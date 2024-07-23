import layers
import torch.nn as nn
#from torch.nn import Parameter


class GNN1(nn.Module):
    def __init__(self, nlayers, n_input, n_hid, n_output, droprate, enable_bias):
        super(GNN1, self).__init__()
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