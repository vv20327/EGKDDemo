# import torch
# import scipy.sparse as sp
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
import random
import os
# import math
from GNN import GNN1
from torch.nn import Parameter
# from sklearn import preprocessing
from sklearn.neighbors import kneighbors_graph


from utils import *
import layers

class GCL(nn.Module):
    def __init__(self, device, nlayers, n_input, n_hid, n_output, droprate, sparse, batch_size, enable_bias):
    #def __init__(self, device, args, feature, n_input, n_hid, n_output, batch_size):
        super(GCL, self).__init__()

        self.embed1 = GNN1(nlayers, n_input, n_hid, n_output, droprate, enable_bias)
        self.embed2 = GNN1(nlayers, n_input, n_hid, n_output, droprate, enable_bias)
        #self.embed1 = LightGCN(args, feature, n_input)#lightGCN
        #self.embed2 = LightGCN(args, feature, n_input)#lightGCN
        self.linear = nn.Linear(n_hid, n_output)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.batch_size = batch_size
        self.device = device

        nn.init.xavier_normal_(self.linear.weight.data)

    # def forward(self, x1, a1, x2, a2, idx_train):
    #     emb1, _ = self.embed1(x1, a1)
    #     emb2, _ = self.embed2(x2, a2)
    #     #获取对比损失
    #
    #     loss = self.batch_nce_loss(emb1, emb2, idx_train)
    #
    #     return loss
    def forward(self, x1, a1, x2, a2, idx_train):
        emb1, _ = self.embed1(x1, a1)
        emb2, _ = self.embed2(x2, a2)
        #获取对比损失

        loss = self.batch_nce_loss(emb1, emb2, idx_train)

        return loss
    #
    def get_emb(self, x, a1):
        emb1, y1 = self.embed1(x, a1)
        # emb2, y2 = self.embed2(x, a2)
        # emb1 = self.embed1(a1)
        # emb2 = self.embed1(a2)
        # emb = 0.3 * emb1 + 0.7 * emb2
        return  emb1
    # def get_emb(self, x, a1, a2):
    #     emb1, y1 = self.embed1(x, a1)
    #     emb2, y2 = self.embed2(x, a2)
    #
    #     emb = 0.3* emb1 + 0.7 * emb2
    #     return  emb
    def get_emb2(self, x, a2):
        emb2, y2 = self.embed2(x, a2)

        return emb2

    #获取对比学习的正负样本
    def set_mask_knn(self, X, k, dataset, metric='cosine'):
        if k != 0:
            path = '../data/knn/{}'.format(dataset)
            if not os.path.exists(path):
                os.makedirs(path)
            file_name = path + '/{}_{}.npz'.format(dataset, k)
            if os.path.exists(file_name):
                knn = sp.load_npz(file_name)
                # print('Load exist knn graph.')
            else:
                print('Computing knn graph...')
                knn = kneighbors_graph(X, k, metric=metric)
                sp.save_npz(file_name, knn)
                print('Done. The knn graph is saved as: {}.'.format(file_name))
            knn = torch.tensor(knn.toarray()) + torch.eye(X.shape[0])
        else:
            knn = torch.eye(X.shape[0])
        self.pos_mask = knn
        self.neg_mask = 1 - self.pos_mask

    def batch_nce_loss(self, z1, z2, idx_train, temperature=0.2, pos_mask=None, neg_mask=None):
        if pos_mask is None and neg_mask is None:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        nnodes = z1.shape[0]
        if (self.batch_size == 0) or (self.batch_size > nnodes):
            loss_0 = self.infonce(z1[idx_train], z2[idx_train], pos_mask[idx_train][:, idx_train], neg_mask[idx_train][:, idx_train], temperature)
            loss_1 = self.infonce(z2[idx_train], z1[idx_train], pos_mask[idx_train][:, idx_train], neg_mask[idx_train][:, idx_train], temperature)
            loss = (loss_0 + loss_1) / 2.0
        else:
            node_idxs = list(range(nnodes))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, self.batch_size)
            loss = 0
            for b in batches:
                weight = len(b) / nnodes
                loss_0 = self.infonce(z1[b], z2[b], pos_mask[:, b][b, :], neg_mask[:, b][b, :], temperature)
                loss_1 = self.infonce(z2[b], z1[b], pos_mask[:, b][b, :], neg_mask[:, b][b, :], temperature)
                loss += (loss_0 + loss_1) / 2.0 * weight
        return loss

    def infonce(self, anchor, sample, pos_mask, neg_mask, tau):
        pos_mask = pos_mask.to(self.device)
        neg_mask = neg_mask.to(self.device)
        sim = self.similarity(anchor, sample) / tau
        exp_sim = torch.exp(sim) * neg_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()

    def similarity(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        #return h1 @ h2.t()
        return torch.mm(h1, h2.t())

class LightGCN(nn.Module):
    def __init__(self, config, feature, n_input):
        super(LightGCN, self).__init__()
        self.config = config
        self.num_users = n_input

        self.__init_weight(feature)

    def __init_weight(self, feature):

        self.num_users = self.config.n_nodes
        self.latent_dim = self.config.n_hid
        self.n_layers = self.config.nlayers
        self.keep_prob = self.config.droprate
        self.A_split = self.config.A_split
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_user.weight.data.copy_(feature)
        print('use pretarined data')
        self.f = nn.Sigmoid()
        #print(f"lgn is already to go(dropout:{self.config.droprate})")
        print("lgn is already to go(dropout:{self.config.droprate})")
        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob, Graph):
        if self.A_split:
            graph = []
            for g in Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(Graph, keep_prob)
        return graph

    def computer(self, g):

        # propagate methods for lightGCN
        users_emb = self.embedding_user.weight
        embs = [users_emb]
        if self.config.droprate:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob, g)
            else:
                g_droped = g
        else:
            g_droped = g

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], users_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                users_emb = users_emb.float()
                all_emb = torch.sparse.mm(g_droped, users_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        # 计算每个用户和物品的嵌入向量，它是通过对 embs 张量的第一个维度（不同层的嵌入）进行均值池化得到的。
        light_out = torch.mean(embs, dim=1)
        return light_out, embs

    def forward(self, g):
        # compute embedding
        all_users, embs = self.computer(g)
        return all_users

class GCN(nn.Module):
    def __init__(self, nlayers, n_input, n_hid, n_output, droprate, enable_bias):
        super(GNN, self).__init__()
        self.graph_convs = nn.ModuleList()
        self.K = nlayers
        if nlayers >= 2:
            self.graph_convs.append(layers.GraphConv(in_features=n_input, out_features=n_hid, bias=enable_bias))
            for k in range(1, nlayers-1):
                self.graph_convs.append(layers.GraphConv(in_features=n_hid, out_features=n_hid, bias=enable_bias))
            self.graph_convs.append(layers.GraphConv(in_features=n_hid, out_features=n_output, bias=enable_bias))
        else:
            self.graph_convs.append(layers.GraphConv(in_features=n_input, out_features=n_output, bias=enable_bias))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, filter):
        if self.K >= 2:
            for k in range(self.K-1):
                x = self.graph_convs[k](x, filter)
                x = self.relu(x)
                x = self.dropout(x)
            x = self.graph_convs[-1](x, filter)
        else:
            x = self.graph_convs[0](x, filter)
        y_pred = self.log_softmax(x)

        return x, y_pred

class GNN_Classifier(nn.Module):
    def __init__(self, ninput, nclass, dropout):
        super(GNN_Classifier, self).__init__()

        self.mlp = nn.Linear(ninput, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x