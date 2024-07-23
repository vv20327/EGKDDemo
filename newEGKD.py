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


from newdataloader1 import *
from model import *
from utils import *


def perturb_graph(features, adj, fea_mask_rate, edge_dropout_rate, args):

    # 删除节点特征
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * fea_mask_rate), replace=False)
    mask[:, samples] = 1
    # if torch.cuda.is_available():
    # mask = mask.cuda()
    features_1 = features * (1 - mask)

    # 删除边
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
def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for multiple GPUs
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_Model(cl_model, optimizer_cl, baseline_classifier, features_1,features_2, adj_1, adj_2):#optimizer_baseline,
    cl_model.train()
    baseline_classifier.eval()

    emb1 = cl_model.get_emb(features_1, adj_1)
    emb2 = cl_model.get_emb2(features_2, adj_2)
   # print(emb1[0])
    pred1 = baseline_classifier(emb1)
    pred2 = baseline_classifier(emb2)
    log_pred1 = F.log_softmax(pred1, dim=1)
    log_pred2 = F.softmax(pred2, dim=1)

    #print(log_pred1)
    #print(len(log_pred2[0]))
    kl_loss = F.kl_div(log_pred1, log_pred2, reduction='batchmean')

    kl_loss.backward()
    optimizer_cl.step()

    return kl_loss.item()#kl_loss loss.item(),

def train_inter(cl_model, optimizer_cl, baseline_classifier, features_1,features_2, adj_1, adj_2):#optimizer_baseline,
    cl_model.train()
    baseline_classifier.eval()

    emb1 = cl_model.get_emb(features_1, adj_1)
    emb2 = cl_model.get_emb2(features_2, adj_2)
    norm1 = (emb1 - emb1.mean(dim=1, keepdim=True)) / emb1.std(1, keepdim=True)
    norm2 = (emb2 - emb2.mean(dim=1, keepdim=True)) / emb2.std(1, keepdim=True)
    corr1 = torch.mul(norm1, norm1)#lp
    corr2 = torch.mul(norm2, norm2)#hp
    corr = torch.mm(norm1, norm2.t())

    loss_s = F.mse_loss(corr1, corr2)
    I = torch.eye(corr.shape[0])
    loss_f = F.mse_loss(corr, I)
    loss = 0.1*loss_s + 0.1*loss_f
    loss.backward()
    optimizer_cl.step()
    #optimizer_baseline.step()

    return loss.item()

def train_GCNModel(baseline_classifier, optimizer_baseline, cl_model, optimizer_cl, features_1, label, idx_train, adj_1, cur_split):
    baseline_classifier.train()
    cl_model.train()

    # get embedding data
    embedding = cl_model.get_emb(features_1, adj_1)

    output = baseline_classifier(embedding)
    loss_train = F.cross_entropy(output[idx_train[:, cur_split]], label[idx_train[:,cur_split]].long())#计算交叉熵损失，主损失

    optimizer_baseline.zero_grad()
    loss_train.backward(retain_graph=True)
    optimizer_baseline.step()
    optimizer_cl.step()
    return loss_train

def main(args):
    torch.cuda.empty_cache()#
    #feature, weights_lp, weights_hp, label, idx_train,  idx_test, n_feat, n_class = load_data()
    feature,feature1, weights_lp, label,label1, idx_train,  idx_test, n_feat, n_class,n_feat1,weights_lp1 = load_data()

    print(idx_train)
    model_dir = './model_save/DGA-2000'
    model_class_dir = './model_save/classifier/DGA-2000'
    model_save_path = 'best_model1.pth'
    emb_save = 'emb1.pth'
    model_path = os.path.join(model_dir, model_save_path)
    classifier_path = os.path.join(model_class_dir, model_save_path)
    emb_path = os.path.join(model_dir, emb_save)
    if torch.cuda.is_available():
        print("GPU")
        device = torch.cuda.set_device(0)
    else:
        print("CPU")
        device = torch.device('cpu')
    print(weights_lp)
    weights_lp = sp.coo_matrix(weights_lp)
    weights_lp1 = sp.coo_matrix(weights_lp1)
    #weights_hp = sp.coo_matrix(weights_hp)

    adj_lp = get_adj(weights_lp, 'sym_renorm_adj')#L
    adj_lp1 = get_adj(weights_lp1, 'sym_renorm_adj')
    #adj_hp = get_adj(weights_hp, 'sym_renorm_adj')#L
    #print(adj_hp)
    if sp.issparse(feature):#
        feature = cnv_sparse_mat_to_coo_tensor(feature, device)
    else:
        feature = feature.to(device)
    if sp.issparse(feature1):
        feature1 = cnv_sparse_mat_to_coo_tensor(feature1, device)
    else:
        feature1 = feature1.to(device)

    adj_lp = cnv_sparse_mat_to_coo_tensor(adj_lp, device)#将这些张量转换为一个稀疏矩阵
    adj_lp1 = cnv_sparse_mat_to_coo_tensor(adj_lp1, device)
    #adj_hp = cnv_sparse_mat_to_coo_tensor(adj_hp, device)
    features_1, adj_1 = perturb_graph(feature, adj_lp, 0.4, 0.07, args)
    features_2, adj_2 = perturb_graph(feature, adj_lp, 0.2, 0.1, args)

    label = label.to(device)
    label1 = label1.to(device)
    idx_train = idx_train.to(device)
    idx_test = idx_test.to(device)
    results, f1_result, auc_result, pre_result = [], [], [], []
    best = 0

    for trial in range(args.ntrials):
        set_env(trial)

        cl_model = GCL(device, nlayers=args.nlayers, n_input= n_feat, n_hid=args.n_hid, n_output=int(args.n_hid/2),droprate=args.droprate, sparse=args.sparse,batch_size=args.cl_batch_size,enable_bias=args.enable_bias).to(device)
        optimizer_cl = torch.optim.Adam(cl_model.parameters(), lr=args.lr_clas, weight_decay=args.w_decay)
        baseline_classifier = GNN_Classifier(ninput=int(args.n_hid/2),
                                             nclass=len(torch.unique(label)), dropout=args.droprate).to(device)
        optimizer_baseline = torch.optim.Adam(baseline_classifier.parameters(),
                                              lr=args.lr_gcl, weight_decay=args.w_decay)

        cur_split = 0 if (idx_train.shape[1] == 1) else (trial % idx_train.shape[1])
        best_acc, best_f1, best_auc, best_pre = 0, 0, 0, 0
        test_idx, train_idx = [], []
        best_split = 0

        for epoch in range(1, args.epochs + 1):#共进行了epochs轮
            train_time_list = []
            train_epoch_begin_time = time.perf_counter()#记录起始时间

            kl_loss = train_Model(cl_model, optimizer_cl, baseline_classifier, features_1,features_2, adj_1, adj_2)
            #loss, kl_loss = train_Model(cl_model, optimizer_cl, baseline_classifier, feature, adj_lp, adj_hp)#optimizer_baseline,
            loss = train_inter(cl_model, optimizer_cl, baseline_classifier, features_1,features_2, adj_1, adj_2)
            class_loss = train_GCNModel(baseline_classifier, optimizer_baseline, cl_model, optimizer_cl, features_1, label, idx_train, adj_1, cur_split)#分类损失，主损失
            emb = cl_model.get_emb(features_1, adj_1)
            output = baseline_classifier(emb)
            acc_train, f1_train, precision_train, auc_train = calc_accuracy(output[idx_train[:, cur_split]], label[idx_train[:,cur_split]].long())#计算准确率
            train_epoch_end_time = time.perf_counter()#记录结束时间
            train_epoch_time_duration = train_epoch_end_time - train_epoch_begin_time
            train_time_list.append(train_epoch_time_duration)

            print("[TRAIN] Epoch:{:04d} | Loss {:.4f} | KL_Loss {:.4f} | Class loss:{:.4f} | TRAIN ACC:{:.4f} | f1_train:{:.2f}| auc_train:{:.2f}| precision_train:{:.2f}| | Training duration: {:.6f}".\
            #print("[TRAIN] Epoch:{:04d} | Loss {:.4f} | Class loss:{:.4f} | TRAIN ACC:{:.4f} | f1_train:{:.2f}| auc_train:{:.2f}| precision_train:{:.2f}| | Training duration: {:.6f}".\
                  #format(epoch, loss, class_loss, acc_train, f1_train, auc_train, precision_train, train_epoch_time_duration))
                  format(epoch, loss, kl_loss, class_loss, acc_train, f1_train, auc_train, precision_train, train_epoch_time_duration))

            if epoch % args.eval_freq == 0 and epoch > 100:
                 cl_model.eval()
                 baseline_classifier.eval()
                 embedding = cl_model.get_emb(feature1, adj_lp1)
                 with torch.no_grad():
                    output = baseline_classifier(embedding)

                    loss_test = F.cross_entropy(output[idx_test[:, cur_split]],
                                                label1[idx_test[:, cur_split]].long())
                    acc_test, f1_test, precision_test, auc_test = calc_accuracy(output[idx_test[:, cur_split]], label1[idx_test[:, cur_split]].long())

                 print(
                     '[TEST] Epoch:{:04d} | Main loss:{:.4f} | TEST ACC:{:.4f} | f1_test:{:.2f}| auc_test:{:.2f}| precision_test:{:.2f}|'
                         .format(epoch, loss_test.item(), acc_test, f1_test, auc_test, precision_test))
                 if f1_test > best_f1:
                     if f1_test > best:
                         best = f1_test
                         best_emb = embedding
                         torch.save(best_emb, emb_path)
                         save_checkpoint(baseline_classifier, classifier_path)
                         save_checkpoint(cl_model, model_path)
                     best_acc = acc_test
                     best_f1 = f1_test
                     best_auc = auc_test
                     best_pre = precision_test
                     best_split = cur_split

        results.append(best_acc)
        f1_result.append(best_f1)
        auc_result.append(best_auc)
        pre_result.append(best_pre)

    test_idx.append(idx_test[:, best_split].numpy())
    np.save('./model_save/DGA-200/best_idx.txt', test_idx)
    train_idx.append(idx_train[:, best_split])
    results = np.array(results, dtype=np.float32)
    f1_result = np.array(f1_result, dtype=np.float32)
    auc_result = np.array(auc_result, dtype=np.float32)
    pre_result = np.array(pre_result, dtype=np.float32)
    print('\n[FINAL RESULT] Dataset:{} | Run:{} | ACC:{:.2f}+-{:.2f} | f1:{:.2f}+-{:.2f} | AUC:{:.2f}+-{:.2f} | Precision:{:.2f}+-{:.2f}'
            .format(args.dataset, args.ntrials, np.mean(results), np.std(results), np.mean(f1_result), np.std(f1_result),
                    np.mean(auc_result), np.std(auc_result), np.mean(pre_result), np.std(pre_result)))

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()#
    parser.add_argument('-ntrials', type=int, default=10)
    parser.add_argument('-eval_freq', type=int, default=20)#400
    parser.add_argument('-epochs', type=int, default=800)#400
    parser.add_argument('-lr_gcl', type=float, default=0.001)
    parser.add_argument('-lr_clas', type=float, default=0.001)#学习率
    parser.add_argument('-w_decay', type=float, default=0.0001)#权重衰减
    parser.add_argument('-droprate', type=float, default=0.1)#掉率
    parser.add_argument('-sparse', type=int, default=1)
    parser.add_argument('-dataset', type=str, default='DGA-200')
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--A_split', type=bool, default=False, help='default as False')

    #GCN Module - Hyper-param
    parser.add_argument('-nlayers', type=int, default=2)#表示神经网络中的所有层数
    parser.add_argument('-n_hid', type=int, default=128)#128，表示隐藏层神经元总数
    parser.add_argument('-cl_batch_size', type=int, default=0)

    # parser.add_argument('-ntrials', type=int, default=10)
    # parser.add_argument('-eval_freq', type=int, default=10)#400
    # parser.add_argument('-epochs', type=int, default=600)#400
    # parser.add_argument('-lr_gcl', type=float, default=0.01)
    # parser.add_argument('-lr_clas', type=float, default=0.01)#学习率
    # parser.add_argument('-w_decay', type=float, default=0.0003)#权重衰减
    # parser.add_argument('-droprate', type=float, default=0.1)#掉率
    # parser.add_argument('-sparse', type=int, default=1)
    # parser.add_argument('-dataset', type=str, default='DGA-2000')
    # parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    # parser.add_argument('--A_split', type=bool, default=False, help='default as False')
    #
    # # GCN Module - Hyper-param
    # parser.add_argument('-nlayers', type=int, default=2)#表示神经网络中的所有层数
    # parser.add_argument('-n_hid', type=int, default=128)#128，表示隐藏层神经元总数
    # parser.add_argument('-cl_batch_size', type=int, default=0)


    args = parser.parse_args()
    print(args)

    main(args)
