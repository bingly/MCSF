import torch
import torch.nn as nn

def cos_sim(A, B):
    # 计算 A 和 B 之间的余弦相似度矩阵
    A_norm = torch.norm(A, dim=1, keepdim=True)  # 计算 A 中每个向量的 L2 范数
    B_norm = torch.norm(B, dim=1, keepdim=True)  # 计算 B 中每个向量的 L2 范数
    similarity_matrix = torch.matmul(A, B.T) / (torch.matmul(A_norm, B_norm.T) + 1e-8)  # 计算余弦相似度
    return similarity_matrix

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(dims)+1):
            if i == 0:
                self.encoder.add_module('Linear%d' % i,  nn.Linear(input_dim, dims[i]))
            elif i == len(dims):
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], feature_dim))
            else:
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            self.encoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


class AutoDecoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoDecoder, self).__init__()
        self.decoder = nn.Sequential()
        dims = list(reversed(dims))
        for i in range(len(dims)+1):
            if i == 0:
                self.decoder.add_module('Linear%d' % i,  nn.Linear(feature_dim, dims[i]))
            elif i == len(dims):
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], input_dim))
            else:
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            self.decoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.decoder(x)


class MCSFNetwork(nn.Module):
    def __init__(self, num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters, batch_size):
        super(MCSFNetwork, self).__init__()
        self.num_views = num_views
        self.num_clusters = num_clusters
        self.batch_size = batch_size

        self.encoders = nn.ModuleList([
            AutoEncoder(input_sizes[idx], dim_high_feature, dims) for idx in range(num_views)
        ])
        self.decoders = nn.ModuleList([
            AutoDecoder(input_sizes[idx], dim_high_feature, dims) for idx in range(num_views)
        ])

        self.label_learning_module = nn.Sequential(
            nn.Linear(dim_high_feature, dim_low_feature),
            nn.Linear(dim_low_feature, num_clusters),
            nn.Softmax(dim=1)
        )

        self.prob_fusion = nn.Parameter(1e-4 * torch.ones(batch_size, num_clusters, dtype=torch.float32),
                                        requires_grad=True)

        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mse_loss = nn.MSELoss()

    def forward(self, data_views):
        lbps, dvs, features, simWs = [], [], [], []

        for idx in range(self.num_views):
            data_view = data_views[idx]
            high_features = self.encoders[idx](data_view)  # Z - the output of encoder
            pairwise_dist = torch.cdist(high_features, high_features)  # [n, n]
            sigma = 1
            simW = torch.exp(-pairwise_dist ** 2 / (2 * sigma ** 2))

            label_probs = self.label_learning_module(high_features)
            label_probs = self.orthogonal_constraint(label_probs)
            data_view_recon = self.decoders[idx](high_features)

            features.append(high_features)
            lbps.append(label_probs)
            dvs.append(data_view_recon)
            simWs.append(simW)

        # self.prob_fusion = sum(lbps) / self.num_views
        self.prob_fusion = nn.Parameter((sum(lbps) / self.num_views).detach())
        return lbps, dvs, self.prob_fusion, simWs

    def loss(self, sub_data_views, lbps, dvs, simWs, adj_graph, temperature_l, alpha, beta):
        criterion = torch.nn.MSELoss()
        loss_list = list()
        for i in range(self.num_views):
            loss_list.append(alpha * self.info_nec_loss_spectral_fusion(lbps[i], self.prob_fusion, adj_graph, temperature_l))
            loss_list.append(beta * self.spectral_loss(simWs[i], lbps[i]))
            loss_list.append(criterion(sub_data_views[i], dvs[i]))
        loss = sum(loss_list)
        return loss

    def orthogonal_constraint(self, P):
        """通过Cholesky分解实现正交约束 P^T P = I"""
        PtP = torch.matmul(P.t(), P)  # [n_clusters, n_clusters]
        L = torch.cholesky(PtP + 1e-4 * torch.eye(P.size(1), device=P.device))
        L_inv = torch.inverse(L)
        P_orth = torch.matmul(P, L_inv.t())  # P_orth^T P_orth = I
        return P_orth

    def spectral_loss(self, W, P):
        """计算谱嵌入损失 L_spc = sum(W_{i,j} * ||P_i - P_j||^2) / n"""
        pairwise_dist = torch.cdist(P, P)  # [n, n]
        loss = torch.sum(W * pairwise_dist ** 2) / P.shape[0]
        return loss

    def info_nec_loss_spectral_fusion(self, F_v, F_consensus, adj_graph, temperature):
        """
        谱嵌入的对比对齐损失
        Args:
            F_v: 视图的谱嵌入
            F_consensus: 共识谱嵌入
            adj_graph: 邻接图 (positive_graph, negative_graph)
            temperature: 温度参数
        """
        # 视图特定嵌入与共识嵌入的对比
        sim_matrix = torch.exp(cos_sim(F_v, F_consensus) / temperature)
        positive_score = torch.sum(sim_matrix * adj_graph[0], dim=1)
        negative_score = torch.sum(sim_matrix * adj_graph[1], dim=1)

        # 确保正值
        positive_score = positive_score[positive_score > 0]
        negative_score = negative_score[negative_score > 0]

        # 计算对比损失
        loss = -(torch.log(positive_score).sum() - torch.log(negative_score).sum())

        # 共识嵌入自身的对比正则化, 对应公式是什么？ 这里参考其他对比学习的代码，分为不同视图，统一视图
        sim_matrix = torch.exp(cos_sim(F_consensus, F_consensus) / temperature)
        sim_matrix = sim_matrix - torch.diag(torch.diag(sim_matrix))
        positive_score = torch.sum(sim_matrix * adj_graph[0], dim=1)
        negative_score = torch.sum(sim_matrix * adj_graph[1], dim=1)

        positive_score = positive_score[positive_score > 0]
        negative_score = negative_score[negative_score > 0]

        consensus_loss = -(torch.log(positive_score).sum() - torch.log(negative_score).sum())

        # F_v的对比损失
        sim_matrix = torch.exp(cos_sim(F_v, F_v) / temperature)
        sim_matrix = sim_matrix - torch.diag(torch.diag(sim_matrix))
        positive_score = torch.sum(sim_matrix * adj_graph[0], dim=1)
        negative_score = torch.sum(sim_matrix * adj_graph[1], dim=1)

        positive_score = positive_score[positive_score > 0]
        negative_score = negative_score[negative_score > 0]

        fv_loss = -(torch.log(positive_score).sum() - torch.log(negative_score).sum())

        return loss + consensus_loss + fv_loss
        # return fv_loss
