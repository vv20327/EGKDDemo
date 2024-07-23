import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch.nn.functional as F

topk = 3


def construt_graph(features, labels, nnodes):

    #dist = cosine_similarity(features)
    #计算节点间相似性
    dist = F.cosine_similarity(features.detach().unsqueeze(1), features.detach().unsqueeze(0), dim=-1)
    weights_lp, weights_hp = torch.zeros((nnodes, nnodes)), torch.zeros((nnodes, nnodes))
    idx_hm, idx_ht = [], []
    #获取前K个相似节点下标
    k1 = 3
    for i in range(dist.shape[0]):
        idx = np.argpartition(dist[i, :], -(k1 + 1))[-(k1 + 1):]
        idx_hm.append(idx)

    counter_hm = 0
    edges_hm = 0

    for i, v in enumerate(idx_hm):
        for Nv in v:
            if Nv == i:
                pass
            else:
                weights_lp[i][Nv] = dist[i][Nv]
                if weights_lp[Nv][i] == 0:
                    edges_hm += 1
                if weights_lp[Nv][i] == 0 and labels[Nv] != labels[i]:
                    counter_hm += 1




    k2 = 3
    for i in range(dist.shape[0]):
        idx = np.argpartition(dist[i, :], k2)[:k2]
        idx_ht.append(idx)

    counter_ht = 0
    edges_ht = 0
    for i, v in enumerate(idx_ht):
        for Nv in v:
            if Nv == i:
                pass
            else:
                weights_hp[i][Nv] = dist[i][Nv]
                if weights_hp[Nv][i] == 0:
                    edges_ht += 1
                if weights_hp[Nv][i] == 0 and labels[Nv] == labels[i]:
                    counter_ht += 1




    ht_error = counter_ht / edges_ht
    hm_error = counter_hm /edges_hm
    print(edges_ht)
    print(edges_hm)
    print(ht_error, hm_error)


    return weights_lp, weights_hp

def load_data():

    data = np.loadtxt('./data/ieee.csv', dtype=float)

    data1 = np.loadtxt('./data/github.csv', dtype=float)
    #data = np.unique(data, axis=0)

    #np.savetxt('./data/github/github.txt', data, fmt='%.6f')


    feature = data[:, :-1]
    feature1 = data1[:, :-1]

    #归一化
    # feature = np.where(feature != 0, np.log2(feature), 0)
    #feature = np.where(feature != 0, feature, 1e-10)
    #feature = np.log2(feature)
    #feature = preprocessing.MinMaxScaler().fit_transform(feature)
    #feature = np.where(feature != 0,feature,0.01)
    label = data[:, -1]
    label1 = data1[:, -1]
    label = torch.tensor(label-1)
    label1 = torch.tensor(label1-1)
    #label = torch.where(label > 3, torch.tensor(0), torch.tensor(1))
    feature = torch.tensor(feature)
    feature1 = torch.tensor(feature1)
    [nnodes, n_feat] = feature.shape
    [nnodes1, n_feat1] = feature1.shape
    n_class = len(torch.unique(label))
    weights_lp, weights_hp = construt_graph(feature, label, nnodes)
    weights_lp1, weights_hp1 = construt_graph(feature1, label1, nnodes1)
    #idx_train, idx_test = get_split(feature, label)
    """
    idx_train = np.load('./data/ieee/train_mask.npy')
    idx_test = np.load('./data/ieee/test_mask.npy')
    idx_train = torch.tensor(idx_train)
    idx_test = torch.tensor(idx_test)"""
    # idx_train = np.load('./new/new0.95/train_mask95.npy')
    # idx_test = np.load('./new/new0.95/test_mask95.npy')
    #0.5 2000
    # idx_train = np.load('./new/train_mask.npy_52.npy')
    # idx_test = np.load('./new/test_mask.npy_52.npy')
    idx_train =np.arange(201)
    idx_test = np.arange(2321)
    # idx_train =np.arange(2321)
    # idx_test = np.arange(201)
    # idx_train = np.load('./new/train_mask2.npy')
    # idx_test = np.load('./new/test_mask2.npy')

    # idx_train = np.load('./new/new0.95/train_mask950.npy')
    # idx_test = np.load('./new/new0.95/test_mask950.npy')
    idx_train = idx_train.reshape( -1,1)
    idx_test = idx_test.reshape(-1,1)
    idx_train = torch.tensor(idx_train)
    idx_test = torch.tensor(idx_test)
    print(idx_train.shape,idx_test.shape)


    return feature,feature1, weights_lp, weights_hp, label,label1, idx_train,  idx_test, n_feat, n_class,n_feat1,weights_lp1
    # return feature, weights_lp, weights_hp, label, idx_train,  idx_test, n_feat, n_class
