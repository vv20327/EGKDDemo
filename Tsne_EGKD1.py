import argparse
import os

import random
import time


# import numpy as np
#
# import scipy.sparse as sp
#
#
# import torch
#
# import torch.nn as nn
import torch.optim as optim

# from sklearn import metrics

from dataloader import *
from model import *
from utils import *


def perturb_graph(features, adj, fea_mask_rate, edge_dropout_rate, args):

    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * fea_mask_rate), replace=False)
    mask[:, samples] = 1

    features_1 = features * (1 - mask)

    if not args.sparse:
        adj_1 = F.dropout(adj, p=edge_dropout_rate, training=True)
    else:
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()

        # Generate mask for dropout
        edge_mask = (torch.rand_like(values) > edge_dropout_rate).float()

        # Apply dropout to non-zero elements
        values = values * edge_mask

        # Create new sparse tensor
        adj_1 = torch.sparse.FloatTensor(indices, values, adj.size())
        #adj_1 = torch.sparse.SpareTensor(indices, values, adj.size())
        #adj_l = cnv_sparse_mat_to_coo_tensor(values, adj.size())#将这些张量转换为一个稀疏矩阵

    return features_1, adj_1
def main(args):
    torch.cuda.empty_cache()
    feature, weights_lp, weights_hp, label, idx_train,  idx_test, n_feat, n_class = load_data()#使用load_data()对数据进行读取，并存储在相应的变量

    print(idx_train)
    model_dir = './model_save/DGA-2000'
    model_class_dir = './model_save/classifier/DGA-2000'
    model_save_path = 'best_model_KDGA1.pth'
    emb_save = 'emb_KDGA1.pth'
    # model_dir = './model_save/DGA-200'
    # model_class_dir = './model_save/classifier/DGA-200'
    # model_save_path = 'best_model_KDGA2.pth'
    # emb_save = 'emb_KDGA2.pth'
    model_path = os.path.join(model_dir, model_save_path)#
    classifier_path = os.path.join(model_class_dir, model_save_path)#
    emb_path = os.path.join(model_dir, emb_save)
    if torch.cuda.is_available():
        print("GPU")
        device = torch.cuda.set_device(0)
    else:
        print("CPU")
        device = torch.device('cpu')
    print(weights_lp)
    weights_lp = sp.coo_matrix(weights_lp)

    weights_hp = sp.coo_matrix(weights_hp)

    adj_lp = get_adj(weights_lp, 'sym_renorm_adj')#L
    #adj_hp = get_adj(weights_hp, 'sym_renorm_adj')#L
    adj_hp = get_adj(weights_hp, 'sym_renorm_lap')

    #print(adj_hp)
    if sp.issparse(feature):
        feature = cnv_sparse_mat_to_coo_tensor(feature, device)
    else:
        feature = feature.to(device)

    adj_lp = cnv_sparse_mat_to_coo_tensor(adj_lp, device)
    adj_hp = cnv_sparse_mat_to_coo_tensor(adj_hp, device)
    label = label.to(device)

    idx_test = idx_test.to(device)

    print(111111111111111111111111111111111111111111111111111111111111111111111111111)
    for epoch in range(10):

        idx = np.arange(0,feature.shape[0])
        print(idx.shape)
        print(idx_test[:,epoch].shape)
        cl_model = GCL(device, nlayers=args.nlayers, n_input= n_feat, n_hid=args.n_hid, n_output=int(args.n_hid/2),droprate=args.droprate, sparse=args.sparse,batch_size=args.cl_batch_size,enable_bias=args.enable_bias).to(device)
        optimizer_cl = torch.optim.Adam(cl_model.parameters(), lr=args.lr_clas, weight_decay=args.w_decay)

        # make_graph_with_tSNE(feature, label, cl_model, adj_lp, adj_hp, model_path, idx)
        make_test_tSNE_KDGA(feature, label, cl_model, adj_lp, adj_hp, model_path, idx_test[:,epoch])
        # make_test_tSNE_KDGA1(feature, label, cl_model, adj_1, adj_2, model_path, idx_test[:,epoch])
    print(111111111111111111111111111111111111111111111111111111111111111111111111111)


    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()#可以理解为添加参数并且管理参数的一个容器

    # ESSENTIAL
    #DGA200 参数
    # parser.add_argument('-ntrials', type=int, default=10)
    # parser.add_argument('-eval_freq', type=int, default=10)#400
    # parser.add_argument('-epochs', type=int, default=800)#400
    # parser.add_argument('-lr_gcl', type=float, default=0.001)
    # parser.add_argument('-lr_clas', type=float, default=0.001)#学习率
    # parser.add_argument('-w_decay', type=float, default=0.0001)#
    # parser.add_argument('-droprate', type=float, default=0.2)#掉率0.2
    # parser.add_argument('-sparse', type=int, default=1)
    # parser.add_argument('-dataset', type=str, default='DGA-200')
    # parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    # parser.add_argument('--A_split', type=bool, default=False, help='default as False')
    # #
    # # GCN Module - Hyper-param
    # parser.add_argument('-nlayers', type=int, default=2)#表示神经网络中的所有层数
    # parser.add_argument('-n_hid', type=int, default=128)#128，表示隐藏层神经元总数
    # parser.add_argument('-cl_batch_size', type=int, default=0)
    #DGA2000参数
    parser.add_argument('-ntrials', type=int, default=10)
    parser.add_argument('-eval_freq', type=int, default=20)#400
    parser.add_argument('-epochs', type=int, default=400)#400
    parser.add_argument('-lr_gcl', type=float, default=0.01)
    parser.add_argument('-lr_clas', type=float, default=0.001)#学习率
    parser.add_argument('-w_decay', type=float, default=0.0004)#权重衰减0.0004
    parser.add_argument('-droprate', type=float, default=0.1)#掉率
    parser.add_argument('-sparse', type=int, default=1)
    parser.add_argument('-dataset', type=str, default='DGA-2000')
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--A_split', type=bool, default=False, help='default as False')

    # GCN Module - Hyper-param
    parser.add_argument('-nlayers', type=int, default=3)#表示神经网络中的所有层数
    parser.add_argument('-n_hid', type=int, default=256)#128，表示隐藏层神经元总数
    parser.add_argument('-cl_batch_size', type=int, default=0)
    args = parser.parse_args()
    print(args)

    main(args)
