import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from adjustText import adjust_text
#from matplotlib.lines import Line2D
from sklearn.metrics import f1_score, roc_auc_score, precision_score
from sklearn.metrics import silhouette_score
plt.rcParams['axes.unicode_minus'] = False

EOS = 1e-10


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def get_adj(dir_adj, gso_type):
    if sp.issparse(dir_adj):

        id = sp.identity(dir_adj.shape[0], format='csc')#id为对角线元素为1的单位矩阵

        # Symmetrizing an adjacency matrix
        adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
        # adj = 0.5 * (dir_adj + dir_adj.transpose())
        if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
                or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
            adj = adj + id

        if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
                or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            row_sum = adj.sum(axis=1).A1
            row_sum_inv_sqrt = np.power(row_sum, -0.5)
            #print(np.isinf(row_sum_inv_sqrt))
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
            deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
            # A_{sym} = D^{-0.5} * A * D^{-0.5}
            sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

            if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
                sym_norm_lap = id - sym_norm_adj
                gso = sym_norm_lap

            else:
                gso = sym_norm_adj

        elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
                or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            row_sum = adj.sum(axis=1).A1
            row_sum_inv = np.power(row_sum, -1)
            row_sum_inv[np.isinf(row_sum_inv)] = 0.
            deg_inv = sp.diags(row_sum_inv, format='csc')

            rw_norm_adj = deg_inv.dot(adj)

            if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
                rw_norm_lap = id - rw_norm_adj
                gso = rw_norm_lap
            else:
                gso = rw_norm_adj

        else:
            #raise ValueError(f'{gso_type} is not defined.')
            raise ValueError('{gso_type} is not defined.')
    else:
        id = np.identity(dir_adj.shape[0])
        # Symmetrizing an adjacency matrix
        adj = np.maximum(dir_adj, dir_adj.T)
        # adj = 0.5 * (dir_adj + dir_adj.T)

        if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
                or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
            adj = adj + id

        if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
                or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            row_sum = np.sum(adj, axis=1)
            row_sum_inv_sqrt = np.power(row_sum, -0.5)
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
            deg_inv_sqrt = np.diag(row_sum_inv_sqrt)
            # A_{sym} = D^{-0.5} * A * D^{-0.5}
            sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

            if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
                sym_norm_lap = id - sym_norm_adj
                gso = sym_norm_lap
            else:
                gso = sym_norm_adj

        elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
                or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            row_sum = np.sum(adj, axis=1).A1
            row_sum_inv = np.power(row_sum, -1)
            row_sum_inv[np.isinf(row_sum_inv)] = 0.
            deg_inv = np.diag(row_sum_inv)
            # A_{rw} = D^{-1} * A
            rw_norm_adj = deg_inv.dot(adj)

            if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
                rw_norm_lap = id - rw_norm_adj
                gso = rw_norm_lap
            else:
                gso = rw_norm_adj

        else:
            #raise ValueError(f'{gso_type} is not defined.')
            raise ValueError('{gso_type} is not defined.')

    return gso

def convert_sp_mat_to_sp_tensor(X, device):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    g = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    g = g.coalesce().to(device)
    return g


def normalize_hm_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj += torch.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todense()
    adj = torch.from_numpy(adj)

    return adj


def normalize_ht_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj += torch.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todense()
    adj = torch.from_numpy(adj)
    adj = torch.eye(adj.shape[0]) - adj * 0.1  # I - (D-1/2)A˜(D-1/2) * a

    return adj



#转换数据格式
def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()#将feature矩阵格式转换
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.complex64 or sp_mat.dtype == np.complex128:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.complex64, device=device, requires_grad=False)
    elif sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        #raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')
        raise TypeError('ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

#计算准确性
def calc_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    accuracy = correct / len(labels)

    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted', zero_division=1)

    # output = output.detach().cpu().numpy()
    # one_hot_labels = np.eye(int(np.max(labels))+1)[labels.astype(int)]

    if labels.max() > 1:
        auc = roc_auc_score(labels, F.softmax(output, dim=-1).detach(), average='macro',multi_class='ovr')
    else:
        auc = roc_auc_score(labels, F.softmax(output, dim=-1)[:, 1].detach(), average='macro')

    return accuracy*100, f1*100, precision*100, auc*100

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def make_graph_with_tSNE(features, labels, cl_model, adj_lp, adj_hp, model_path, idx):


    cl_model.load_state_dict(torch.load(model_path))
    embedding = cl_model.get_emb(features, adj_lp, adj_hp)
    features = features[idx]
    embedding = embedding[idx]
    # list = ['LH', 'MOTH', 'MTH', 'HTH', 'PD', 'SD', 'AD']
    list = ['N','LH', 'MTH', 'HTH', 'PD', 'SD', 'AD']
    #list = ['T', 'D']

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=800, random_state=42)
    X_tsne0 = tsne.fit_transform(features)  # features是Cora数据集的特征矩阵
    X_tsne = tsne.fit_transform(embedding.detach().numpy())

    unique_labels = np.unique(labels)


    colors = plt.cm.get_cmap("tab10", len(unique_labels))
    #colors = ['red', 'blue']
    alpha = 1
    maker = ['o', 's', 'D', '*', '+', 'x', 'H']
    size = 10  # 设置每个类别的点大小


    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(X_tsne0[mask, 0], X_tsne0[mask, 1], c=[colors(i)], label=list[i], s=size, marker=maker[i],
                    alpha=alpha)
        plt.legend()
    plt.title('t-SNE Visualization of DGA-200 Dataset(original feature) ')
    plt.show()

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors(i)], label=list[i], s=size, marker=maker[i], alpha=alpha)
        plt.legend()
    plt.title('t-SNE Visualization of DGA-200 Dataset(embedding) ')
    plt.show()

def make_test_tSNE(features, labels, cl_model, adj_lp, adj_hp, model_path, idx):

    labels = labels[idx]
    cl_model.load_state_dict(torch.load(model_path))
    embedding = cl_model.get_emb(features, adj_lp, adj_hp)

    list = ['LH', 'MOTH', 'MTH', 'HTH', 'PD', 'SD', 'AD']
    #list = ['T', 'D']

    test_feature = features[idx]
    test_emb = embedding[idx]
    tsne0 = TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=1000, random_state=42)#
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=50, n_iter=1000, random_state=42)  #
    X_tsne0 = tsne0.fit_transform(test_feature)  # features是Cora数据集的特征矩阵
    X_tsne = tsne.fit_transform(test_emb.detach().numpy())

    unique_labels = np.unique(labels)

    colors = plt.cm.get_cmap("tab1", len(unique_labels))
    #colors = ['red', 'blue']
    alpha = 1
    maker = ['o', 's', 'D', '*', '+', 'x', 'H']
    size = 10


    plt.figure(figsize=(8, 6))
    mask = labels == 4#-1
    # ind = [429, 422, 413, 476]
    # mask[ind] = False
    indices = np.where(mask)[0]
    #plt.scatter(X_tsne0[mask, 0], X_tsne0[mask, 1], c=[colors(4)], label=f'Class {4}', s=size, marker=maker[4],
   #                 alpha=alpha)
    plt.scatter(X_tsne0[mask, 0], X_tsne0[mask, 1], c=[colors(4)], label='Class {4}', s=size, marker=maker[4],alpha=alpha)
    tests = []
    for i, (x, y) in enumerate(zip(X_tsne0[mask, 0], X_tsne0[mask, 1])):
        tests.append(plt.annotate(indices[i], (x, y), fontsize=5, ha='right', va='bottom'))
        #plt.annotate(indices[i], (x, y), fontsize=5, ha='right', va='bottom')
    adjust_text(tests)
    plt.legend()
    """
    legend_elements = []
    for i, label in enumerate(unique_labels):
        legend_elements.append(Line2D([0], [0], color=colors(int(labels[i])), label=label))

    for i in range(len(idx)):
        plt.scatter(X_tsne0[i, 0], X_tsne0[i, 1], c=[colors(int(labels[i]))], s=size,
                    marker=maker[int(labels[i])],
                    alpha=alpha)
    """   #plt.text(X_tsne0[i, 0], X_tsne0[i, 1], int(idx[i]), fontsize=8, ha='right', va='bottom')
    plt.title('t-SNE Visualization of DGA-2000 Dataset(test feature) ')
    plt.show()

    plt.figure(figsize=(8, 6))
    test = []
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors(4)], label='Class {4}', s=size, marker=maker[4],
                    alpha=alpha)

    for i, (x, y) in enumerate(zip(X_tsne[mask, 0], X_tsne[mask, 1])):
        test.append(plt.annotate(indices[i], (x, y), fontsize=7, ha='right', va='bottom'))
        #plt.annotate(indices[i], (x, y), fontsize=7, ha='right', va='bottom')
    adjust_text(test)
    plt.legend()
    """
    for i in range(len(idx)):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=[colors(int(labels[i]))], s=size,#colors[int(labels[i])]
                    marker=maker[int(labels[i])], alpha=alpha)
        #plt.text(X_tsne[i, 0], X_tsne[i, 1], int(idx[i]), fontsize=8, ha='right', va='bottom')
    """
    plt.title('t-SNE Visualization of DGA-200 Dataset(test_embed) ')
    plt.show()
def make_test_tSNE_KDGA(features, labels, cl_model, adj_lp, adj_hp, model_path, idx):

        labels = labels[idx]
        cl_model.load_state_dict(torch.load(model_path))
        embedding = cl_model.get_emb(features, adj_lp, adj_hp)
        # list = ['N','LH', 'MTH', 'HTH', 'PD', 'SD', 'AD']#2000的标签
        list = ['LH', 'MOTH', 'MTH', 'HTH', 'PD', 'SD', 'AD']

        test_feature = features[idx]
        test_emb = embedding[idx]

        silhouette_avg = silhouette_score(test_feature.detach().numpy(), labels)
        silhouette_avg1 = silhouette_score(test_emb.detach().numpy(), labels)
        print(silhouette_avg,"原始轮廓系数")
        print(silhouette_avg1,"KDGA_1系数")
        tsne0 = TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=1000, random_state=42)#
        tsne = TSNE(n_components=2, perplexity=15, learning_rate=50, n_iter=1000, random_state=42)  #
        X_tsne0 = tsne0.fit_transform(test_feature)
        X_tsne = tsne.fit_transform(test_emb.detach().numpy())
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap("tab10", len(unique_labels))
        #colors = ['red', 'blue']
        alpha = 1
        maker = ['o', 's', 'D', '*', '+', 'x', 'H']
        size = 15  # 设置每个类别的点大小


        plt.figure(figsize=(8, 6))
        for i in range(0,4):
             mask = labels == i
             plt.scatter(X_tsne0[mask, 0], X_tsne0[mask, 1], c=[colors(i)], label=list[i], s=size, marker=maker[i],
                        alpha=alpha)
             plt.legend()

        plt.title('t-SNE Visualization of DGA-2000 Dataset(test feature) ')
        plt.savefig('./image/1.png', dpi=500, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(8, 6))

        for i in range(0,4):
             mask = labels == i
             plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors(i)], label=list[i], s=size, marker=maker[i],
                        alpha=alpha)
             plt.legend()
        plt.title('t-SNE Visualization of DGA-2000 Dataset(test_embedding) ')
        plt.savefig('./image/2.png', dpi=500, bbox_inches='tight')
        plt.show()


def make_test_tSNE_KDGA1(features, labels, cl_model, adj_lp, adj_hp, model_path, idx):
    labels = labels[idx]#获取测试数据标签
    cl_model.load_state_dict(torch.load(model_path))#加载模型
    embedding = cl_model.get_emb(features, adj_lp, adj_hp)#根据特征，获取节点嵌入
        # list = ['N','LH', 'MTH', 'HTH', 'PD', 'SD', 'AD']#2000的标签
    list = ['LH', 'MOTH', 'MTH', 'HTH', 'PD', 'SD', 'AD']#200的标签
    test_feature = features[idx]  # 获取测试数据特征
    test_emb = embedding[idx]  # 获取测试数据嵌入

    # 计算轮廓系数
    silhouette_avg = silhouette_score(test_feature.detach().numpy(), labels)
    silhouette_avg1 = silhouette_score(test_emb.detach().numpy(), labels)
    print(silhouette_avg, "原始轮廓系数")
    print(silhouette_avg1, "KDGA_1系数")

    tsne0 = TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=1000, random_state=42)
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=50, n_iter=1000, random_state=42)
    X_tsne0 = tsne0.fit_transform(test_feature)  # 对原始特征进行降维
    X_tsne = tsne.fit_transform(test_emb.detach().numpy())  # 对学习后的嵌入进行降维

    # 获取不同类别的标签
    unique_labels = np.unique(labels)  # 获取其七种标签

    # 创建颜色映射，为每个类别分配不同的颜色
    colors = plt.cm.get_cmap("tab10", len(unique_labels))
    alpha = 1
    maker = ['o', 's', 'D', '*', '+', 'x', 'H']
    size = 15  # 设置每个类别的点大小

    # 绘制降维后的数据，每个类别使用不同颜色和大小
    plt.figure(figsize=(8, 6))

    for i in range(4, 7):  # 循环0到4
        # 随机选择每个类别中50%的数据点
        p = [0.5] * len(unique_labels)
        random_mask = np.random.choice([True, False], labels.sum(), p=p)

        subset_mask = (labels == i) & random_mask

        plt.scatter(X_tsne0[subset_mask, 0], X_tsne0[subset_mask, 1], c=[colors(i)], label=list[i], s=size, marker=maker[i],
                    alpha=alpha)
        plt.legend()

    plt.title('t-SNE Visualization of DGA-2000 Dataset(test feature) ')
    plt.show()

    plt.figure(figsize=(8, 6))

    for i in range(4, 7):  # 循环0到4
        # 随机选择每个类别中50%的数据点
        p = [0.5] * len(unique_labels)
        random_mask = np.random.choice([True, False], labels.sum(), p=p)

        subset_mask = (labels == i) & random_mask

        plt.scatter(X_tsne[subset_mask, 0], X_tsne[subset_mask, 1], c=[colors(i)], label=list[i], s=size, marker=maker[i],
                    alpha=alpha)
        plt.legend()

    plt.title('t-SNE Visualization of DGA-2000 Dataset(test_embedding) ')
    plt.show()

    # Compute silhouette scores


def save_checkpoint(model, model_path):
    torch.save(model.state_dict(), model_path)

def make_loss(epochs,class_loss,kl_loss,loss1):
    plt.plot(range(1, epochs + 1), loss1, label='Loss')
    plt.plot(range(1, epochs + 1), kl_loss, label='kl_loss')
    plt.plot(range(1, epochs + 1), class_loss, label='class_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()
#if __name__ == "__main__":