import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layer import *
import torch
from sklearn.manifold import TSNE

class nof(nn.Module):
    def __init__(self, mydev, adj, nnode, nfeat, nlayers,nhidden, nclass, dropout, variant):
        super(nof, self).__init__()
        self.adj=adj
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(noflayer(nnode, nhidden, nhidden, self.adj, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.in_features = nhidden
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        # self.a = nn.Parameter(torch.empty(size=(2*self.in_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        # self.alpha = 0.2
        # self.leakyrelu = nn.LeakyReLU(self.alpha)
        # self.ind = torch.arange(0, self.adj.shape[0], 1)
        # self.ind = self.ind.reshape((self.ind.shape[0],1))
        # self.zero = torch.zeros((self.adj.shape[0], self.adj.shape[1]))
        # self.zero = self.zero.cuda()
        
    # def attention(self, feature):
    #     #feature = torch.mm(feature, self.weight_matrix_att)
    #     feat_1 = torch.matmul(feature, self.a[:self.in_features, :])
    #     feat_2 = torch.matmul(feature, self.a[self.in_features:, :])
        
    #     # broadcast add
    #     e = feat_1 + feat_2.T
    #     e = self.leakyrelu(e)
        
    #     zero_vec = -9e15*torch.ones_like(e)
    #     att = torch.where(self.adj.to_dense() > 0, e, zero_vec)
    #     att = F.softmax(att, dim=1).clone()
        
    #     raf_2 = att.multinomial(num_samples=self.max_degree, replacement=True)
    #     self.zero[self.ind, raf_2] = 1
    #     zero = torch.where(self.zero > 0, e, zero_vec)
    #     zero = F.softmax(zero, dim=1)
    #     return zero
    
    
    def forward(self, x):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        #zero = self.attention(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, _layers[0], i + 1))
            #_layers.append(layer_inner)
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        # mylast=layer_inner

        # tsne = TSNE(n_components=2)
        # tsne.fit_transform(mylast.detach().cpu().numpy())
        # np.save('coralastlayer.npy',tsne.embedding_)
        return F.log_softmax(layer_inner, dim=1)


class combinemodel(nn.Module):
    def __init__(self, gamma, nnode, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant):
        super(combinemodel, self).__init__()
        self.gamma=gamma
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(combinelayer(nnode, nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, support0, support1, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, support0, support1, adj, self.gamma, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

class GWCNII(nn.Module):
    def __init__(self, mydev, myf, support0, support1, nnode, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant):
        super(GWCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(WaveletConvolution(nnode, nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.support0=support0
        self.support1=support1

    def forward(self, x):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, self.support0, self.support1, _layers[0], self.lamda, self.alpha, i + 1))
        #layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha,variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner


if __name__ == '__main__':
    pass





