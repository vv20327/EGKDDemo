import numpy as np
import torch

from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
import torch.nn.functional as F

topk = 5


def construt_graph(features, labels, nnodes):


    dist = F.cosine_similarity(features.detach().unsqueeze(1), features.detach().unsqueeze(0), dim=-1)#表示相似度矩阵
    #print(dist)
    weights_lp, weights_hp = torch.zeros((nnodes, nnodes)), torch.zeros((nnodes, nnodes))
    idx_hm, idx_ht = [], []

    k1 = 5
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

    print(11111111111111,weights_lp)


    k2 = 5
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


def get_split(features, labels):
    #print(torch.unique(labels))
    n = len(torch.unique(labels))
    labels = labels.unsqueeze(1)

    num_samples = features.shape[0]
    data = torch.cat((features, labels), dim=1)
    trains, tests = [], []
    dx_train, dx_test, dy_train, dy_test = [], [], [], []
    for j in range(10):
        indices_test, data_test = [], []
        for i in range(0, n):#0-6
            temp = data[data[:, -1] == i]
            X = temp[:, :-1]
            y = temp[:, -1]
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            X_train, X_test, y_train, y_test = train_test_split(X[indices], y[indices], test_size=0.15, random_state=42)#测试集大小为0.5
            data_test.append(X_test)
            dy_train.append(y_train.unsqueeze(0))
            dx_train.append(X_train.unsqueeze(0))
            dy_test.append(y_test.unsqueeze(0))
            dx_test.append(X_test.unsqueeze(0))

        data_test = torch.cat(data_test, dim=0)

        for j in range(len(data_test)):
            for i in range(num_samples):
                if i not in indices_test:
                    if all(features[i].eq(data_test[j])):#eq比较大小
                        indices_test.append(i)
                        break
        indices_train = torch.zeros(num_samples, dtype=torch.bool)  # torch.Size([n])
        indices_train.fill_(True)
        indices_train[indices_test] = False
        indices_train = np.where(indices_train)[0]
        indices_test = np.array(indices_test)

        indices_train = torch.tensor(indices_train)
        indices_test = torch.tensor(indices_test)

        trains.append(indices_train.unsqueeze(1))
        tests.append(indices_test.unsqueeze(1))
    train_mask_all = torch.cat(trains, 1)
    test_mask_all = torch.cat(tests, 1)
    print(train_mask_all.shape)
    X_train_mask = torch.cat(dx_train, 1)
    y_train_mask = torch.cat(dy_train, 1)
    X_test_mask = torch.cat(dx_test, 1)
    y_test_mask = torch.cat(dy_test, 1)


    np.save('./new/new0.85/X_train_mask85.npy', X_train_mask)
    np.save('./new/new0.85/y_train_mask85.npy', y_train_mask)
    np.save('./new/new0.85/X_test_mask85.npy', X_test_mask)
    np.save('./new/new0.85/y_test_mask85.npy', y_test_mask)

    np.save('./new/new0.85/train_mask85.npy', train_mask_all)
    np.save('./new/new0.85/test_mask85.npy', test_mask_all)
    return train_mask_all, test_mask_all

def load_data():

    # data = np.loadtxt('./data/ieee.csv', dtype=float)


    data = np.loadtxt('./data/github.csv', dtype=float)
    #data = np.unique(data, axis=0)

    # np.savetxt('./data/github/github.txt', data, fmt='%.6f')
    #

    feature = data[:, :-1]
    print(feature.shape)
    #归一化
    #feature = np.where(feature != 0, np.log2(feature), 0)
    #feature = preprocessing.MinMaxScaler().fit_transform(feature)
    label = data[:, -1]
    #print(label)
    #label = torch.tensor(label-1)
    label = torch.tensor(label-1)
    #print(label)
    #label = torch.where(label > 3, torch.tensor(0), torch.tensor(1))
    feature = torch.tensor(feature)
    [nnodes, n_feat] = feature.shape
    #print(nnodes,n_feat)#201行，5列
    n_class = len(torch.unique(label))

    weights_lp, weights_hp = construt_graph(feature, label, nnodes)
    # idx_train, idx_test = get_split(feature, label)
    #print(idx_train.shape)
    """
    idx_train = np.load('./data/ieee/train_mask.npy')
    idx_test = np.load('./data/ieee/test_mask.npy')
    idx_train = torch.tensor(idx_train)
    idx_test = torch.tensor(idx_test)"""
    idx_train = np.load('./new/train_mask.npy_52.npy')
    idx_test = np.load('./new/test_mask.npy_52.npy')

    idx_train = idx_train.reshape( -1,10)
    idx_test = idx_test.reshape(-1,10)
    idx_train = torch.tensor(idx_train)
    idx_test = torch.tensor(idx_test)
    print(idx_test.shape)
    print(idx_train.shape)

    return feature, weights_lp, weights_hp, label, idx_train,  idx_test, n_feat, n_class
