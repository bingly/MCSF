import os, random, sys

import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset
# from torch.nn.functional import normalize
from utils import *


class MultiviewData(Dataset):
    def __init__(self, db, device, path="../datasets/"):
        self.data_views = list()

        if db == "MSRCv1":
            mat = sio.loadmat(os.path.join(path, 'MSRCv1.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "COIL20":
            mat = sio.loadmat(os.path.join(path, 'COIL20.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])

            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        else:
            mat = sio.loadmat(os.path.join(path, db + '.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                X_data[0, idx] = X_data[0, idx].transpose()
                self.data_views.append(X_data[0, idx].astype(np.float32))
            self.labels = np.array(np.squeeze(mat['gt'])).astype(np.int32)

        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sub_data_views = list()
        for view_idx in range(self.num_views):
            data_view = self.data_views[view_idx]
            sub_data_views.append(data_view[index])

        return sub_data_views, self.labels[index]


def get_multiview_data(mv_data, batch_size):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    # 使用固定的随机种子生成器
    g = torch.Generator()
    g.manual_seed(42)  # 固定值

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=g  # 使用固定的随机生成器
    )

    return mv_data_loader, num_views, num_samples, num_clusters


def get_all_multiview_data(mv_data):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=num_samples,
        shuffle=True,
        drop_last=True,
    )

    return mv_data_loader, num_views, num_samples, num_clusters
