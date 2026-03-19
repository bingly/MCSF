import numpy as np
import os
import math
import random
import torch
from sklearn.metrics import normalized_mutual_info_score, v_measure_score, adjusted_rand_score, accuracy_score
from sklearn import cluster
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import pdist, squareform

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def normalize_multiview_data(data_views, row_normalized=True):
    '''The rows or columns of a matrix normalized '''
    norm2 = Normalizer(norm='l2')
    num_views = len(data_views)
    for idx in range(num_views):
        if row_normalized:
            data_views[idx] = norm2.fit_transform(data_views[idx])
        else:
            data_views[idx] = norm2.fit_transform(data_views[idx].T).T

    return data_views


def spectral_clustering(W, num_clusters):
    """
    Apply spectral clustering on W.
    # Arguments
    :param W: an affinity matrix
    :param num_clusters: the number of clusters
    :return: cluster labels.
    """
    # spectral = cluster.SpectralClustering(n_clusters=num_clusters, eigen_solver='arpack', affinity='precomputed',
    #                                       assign_labels='discretize')

    assign_labels='kmeans'
    spectral = cluster.SpectralClustering(n_clusters=num_clusters, eigen_solver='arpack', affinity='precomputed')
    spectral.fit(W)
    labels = spectral.fit_predict(W)

    return labels


def cal_spectral_embedding(W, num_clusters):

    D = np.diag(1 / np.sqrt(np.sum(W, axis=1) + math.e))
    # D1 = np.diag(np.power((np.sum(W, axis=1) + math.e), -0.5))
    Z = np.dot(np.dot(D, W), D)
    U, _, _ = np.linalg.svd(Z)
    eigenvectors = U[:, 0 : num_clusters]

    return eigenvectors


def cal_spectral_embedding_1(W, num_clusters):
    D = np.diag(np.power((np.sum(W, axis=1) + math.e), -0.5))
    L = np.eye(len(W)) - np.dot(np.dot(D, W), D)
    eigvals, eigvecs = np.linalg.eig(L)
    x_val = []
    x_vec = np.zeros((len(eigvecs[:, 0]), len(eigvecs[0])))
    for i in range(len(eigvecs[:, 0])):
        for j in range(len(eigvecs[0])):
            x_vec[i][j] = eigvecs[i][j].real
    for i in range(len(eigvals)):
        x_val.append(eigvals[i].real)
    # 选择前n个最小的特征向量
    indices = np.argsort(x_val)[: num_clusters]
    eigenvectors = x_vec[:, indices[: num_clusters]]

    return eigenvectors


def cal_l2_distances(data_view):
    '''
    calculate Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        dists[i] = np.sqrt(np.sum(np.square(data_view - data_view[i]), axis=1)).T
    return dists


def cal_l2_distances_1(data_view):
    '''
    calculate Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            dists[i][j] = np.sqrt(np.sum(np.square(data_view[i]-data_view[j])))

    return dists


def cal_squared_l2_distances(data_view):
    '''
    calculate squared Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        dists[i] = np.sum(np.square(data_view - data_view[i]), axis=1).T
    return dists


def cal_squared_l2_distances_1(data_view):
    '''
    calculate squared Euclidean distance
    tips: each row in data_view represents a sample
    '''
    num_samples = data_view.shape[0]
    dists = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            dists[i][j] = np.sum(np.square(data_view[i]-data_view[j]))

    return dists


def cal_similiarity_matrix(data_view, k):
    '''
    calculate similiarity matrix
    '''
    num_samples = data_view.shape[0]
    dist = cal_squared_l2_distances(data_view)

    W = np.zeros((num_samples,num_samples), dtype=float)

    idx_set = dist.argsort()[::1]
    for i in range(num_samples):
        idx_sub_set = idx_set[i, 1:(k+2)]
        di = dist[i, idx_sub_set]
        W[i, idx_sub_set] = (di[k]-di) / (di[k] - np.mean(di[0:(k-1)]) + math.e)

    W = (W + W.T) / 2

    return W

def single_view_adj_graph(index_adj, n):
    adj_graph = np.zeros((n, n))
    for i in range(n):
        adj_graph[index_adj[:, i], i] = 1
    return adj_graph


def adj_graphs(X, n_samples, k, type):
    positive_adj_graphs = []
    negative_adj_graphs = []
    for i in range(len(X)):
        X_cpu = X[i].detach().cpu().numpy()
        if type == "cosine":
            positive_pairs_graph_i = squareform(pdist(X_cpu, 'cosine'))
            index_adj_i = np.argsort(positive_pairs_graph_i, axis=0)
            positive_pairs_graph_i = single_view_adj_graph(index_adj_i[:k, :], n_samples)
            positive_adj_graphs.append(positive_pairs_graph_i)

        if type == "euclidean":
            positive_pairs_graph_i = squareform(pdist(X_cpu, 'euclidean'))
            index_adj_i = np.argsort(positive_pairs_graph_i, axis=0)
            positive_pairs_graph_i = single_view_adj_graph(index_adj_i[:k, :], n_samples)
            positive_adj_graphs.append(positive_pairs_graph_i)

    return positive_adj_graphs

def adj_graphs_via_simWs(simWs, n_samples, k):
    """
    Generate adjacency graphs based on similarity matrices (simWs).
    :param simWs: List of similarity matrices for each view.
    :param n_samples: Number of samples.
    :param k: Number of nearest neighbors.
    :return: List of adjacency graphs for each view.
    """
    positive_adj_graphs = []
    negative_adj_graphs = []

    for simW in simWs:
        # Sort indices based on similarity values
        index_adj = np.argsort(simW.detach().cpu().numpy(), axis=0)  # Descending order for similarity
        positive_pairs_graph = np.zeros((n_samples, n_samples))

        # Construct positive adjacency graph
        for i in range(n_samples):
            positive_pairs_graph[index_adj[:k, i], i] = 1

        positive_adj_graphs.append(positive_pairs_graph)

    return positive_adj_graphs

# 没有移除自连接的
def get_negative_graph(positive_pairs_graph, n_samples):
    negative_pairs_graph = np.ones((n_samples, n_samples))
    negative_pairs_graph[positive_pairs_graph > 0] = 0
    return negative_pairs_graph


def reformulate_positive_graph(positive_pairs_graph, n_samples):
    positive_pairs_graph = positive_pairs_graph - np.eye(n_samples)
    positive_pairs_graph = positive_pairs_graph / np.sum(positive_pairs_graph, axis=0)
    positive_pairs_graph = positive_pairs_graph + np.eye(n_samples)
    return positive_pairs_graph


def fused_adj_graph(positive_adj_graphs, n_samples, n_views):
    fused_positive_pairs_graph = np.zeros((n_samples, n_samples))
    for i in range(n_views):
        fused_positive_pairs_graph = np.maximum(positive_adj_graphs[i], fused_positive_pairs_graph)

    fused_positive_pairs_graph[fused_positive_pairs_graph > 0] = 1
    fused_positive_pairs_graph = reformulate_positive_graph(fused_positive_pairs_graph, n_samples)
    fused_negative_pairs_graph = get_negative_graph(fused_positive_pairs_graph, n_samples)

    adj_graph = np.stack((fused_positive_pairs_graph.T, fused_negative_pairs_graph.T), axis=0)
    return adj_graph

def plot_tsne(data, labels, title, db_name=None):

    # Data standardization
    scaler = StandardScaler()
    total_fusion_scaled = scaler.fit_transform(data)

    # Use t-SNE for dimensionality reduction
    tsne = TSNE(metric='euclidean', random_state=42)
    total_fusion_2d = tsne.fit_transform(total_fusion_scaled)

    # Plot scatter plot
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        plt.scatter(total_fusion_2d[labels == label, 0],
                    total_fusion_2d[labels == label, 1],
                    label=f'Class {label}')
    plt.title(f'{title}')
    plt.xticks([])
    plt.yticks([])


    if db_name:
        save_path = f'./results/{db_name} {title}.png'
        plt.savefig(save_path)
    else:
        plt.show()