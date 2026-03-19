import os
import time
import argparse
import warnings

import utils as U
from layers import *
from metrics import *
from dataprocessing import *

import yaml
import torch
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser(description='MCSF')

    parser.add_argument('--db', type=str, default='MSRCv1',
                        choices=['MSRCv1', '3Sources', 'EYaleB10', 'MNIST_USPS'],
                        help='dataset name')
    parser.add_argument('--seed', type=int, default=10, help='Initializing random seed.')
    parser.add_argument("--con_epochs", default=100, help='Number of epochs to fine-tuning.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='Initializing learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0., help='Initializing weight decay.')
    parser.add_argument("--temperature_l", type=float, default=1.0)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='The total number of samples must be evenly divisible by batch_size.')
    parser.add_argument('--gpu', default='0', type=str, help='GPU device idx.')

    return parser

def train(con_epochs, mv_data_loader, num_views, num_samples, num_clusters, alpha, beta, batch_size, knn=5):
    acc_array = []
    nmi_array = []
    f1_array = []
    ari_array = []
    loss_values = []

    for epoch in range(con_epochs):
        total_fusion = []
        total_lbps = []
        labels_vector = []
        pred_vectors = []

        for v in range(num_views):
            pred_vectors.append([])
            total_lbps.append([])

        for batch_idx, (sub_data_views, sub_labels) in enumerate(mv_data_loader):
            positive_adj_graphs = U.adj_graphs(sub_data_views, batch_size, knn, 'cosine')  # 'euclidean' 'cosine'
            fused_adj_graph = U.fused_adj_graph(positive_adj_graphs, batch_size, num_views)
            adj_graph = torch.tensor(fused_adj_graph, dtype=torch.float32, device=device)

            lbps, dvs, prob_fusion, simWs = MCSF(sub_data_views)
            loss = MCSF.loss(sub_data_views, lbps, dvs, simWs, adj_graph, args.temperature_l, alpha, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                total_fusion.extend(prob_fusion)
                labels_vector.extend(sub_labels)

                for idx in range(num_views):
                    total_lbps[idx].extend(lbps[idx].detach().cpu().numpy())
                    pred_label = torch.argmax(lbps[idx], dim=1)
                    pred_vectors[idx].extend(pred_label.detach().cpu().numpy())

        if epoch % 10 == 0:
            labels_vector = np.array(labels_vector).reshape(num_samples)
            total_fusion = [item.detach().cpu().numpy() for item in total_fusion]

            clustering = KMeans(n_clusters=num_clusters, n_init='auto', random_state=23).fit(total_fusion)
            predict_labels = clustering.labels_
            labels_vector = np.array(labels_vector).reshape(num_samples)
            acc, nmi, ari, f1 = calculate_metrics(labels_vector, predict_labels)
            acc_array.append(acc)
            nmi_array.append(nmi)
            f1_array.append(f1)
            ari_array.append(ari)

            print('Epoch {}: alpha = {} beta = {} KNN = {} Batch = {} ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} F1={:.4f} loss:{:.7f}'.format(epoch, alpha, beta, knn, batch_size, acc, nmi, ari, f1, loss / num_samples))
        loss_values.append(loss.item() / num_samples)

    # plot_tsne(total_fusion, labels_vector, title='', db_name=args.db)

    return acc_array, nmi_array, f1_array, ari_array, loss_values

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # load config file
    config_file =  f'./config/{args.db}.yaml'
    with open(config_file) as f:
        if hasattr(yaml, "FullLoader"):
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        else:
            configs = yaml.load(f.read())

    args = vars(args)
    args.update(configs)
    args = argparse.Namespace(**args)

    print("==========\nArgs:{}\n==========".format(args))

    # torch.cuda.setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mv_data = MultiviewData(args.db, device)
    input_sizes = np.zeros(mv_data.num_views, dtype=int)
    for idx in range(mv_data.num_views):
        input_sizes[idx] = mv_data.data_views[idx].shape[1]

    U.set_seed(args.seed)
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, args.batch_size)

    # initialize MCSF network
    MCSF = MCSFNetwork(num_views, input_sizes, args.dims, args.dim_high_feature, args.dim_low_feature, num_clusters,
                        args.batch_size).to(device)
    optimizer = torch.optim.Adam(MCSF.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    t = time.time()

    acc_array, nmi_array, f1_array, ari_array, loss_values = train(args.con_epochs, mv_data_loader, num_views,
                                                                    num_samples, num_clusters, args.alpha, args.beta, args.batch_size, args.knn)
    exec_time = time.time() - t
    print("Total time elapsed: {}s".format(exec_time))

    best_index = np.argmax(np.array(acc_array))
    acc_max = acc_array[best_index] * 100
    nmi_max = nmi_array[best_index] * 100
    f1_max = f1_array[best_index] * 100
    ari_max = ari_array[best_index] * 100
    print(
        'MCSF,{:.2f},{:.2f},{:.2f},{:.2f},{:.6f} alpha = {} beta = {} KNN = {} Batch = {} '.format(
            acc_max, nmi_max, ari_max, f1_max, exec_time, args.alpha, args.beta, args.knn, args.batch_size))
    with open('./results/res_%s.txt' % args.db, 'a+') as f:
        f.write(
            '{} \t {} \t {} \t {} \t {} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.4f} \t {} \t {} \t {}  \n'.format(
                args.dim_high_feature, args.dim_low_feature, args.seed, args.batch_size,
                args.learning_rate, acc_max, nmi_max, ari_max, f1_max, (time.time() - t), args.alpha, args.beta, args.knn))
        f.write(
            'MCSF,{:.2f},{:.2f},{:.2f},{:.2f},{:.6f} alpha = {} beta = {} KNN = {} Batch = {} '.format(
                acc_max, nmi_max, ari_max, f1_max, exec_time, args.alpha, args.beta, args.knn, args.batch_size))
        f.flush()
