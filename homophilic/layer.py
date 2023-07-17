from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from scipy import sparse 
import numpy as np
from torch.nn.functional import normalize

class noflayer(nn.Module):
    def __init__(self, nnode, in_features, out_features, adj, residual=False, variant=False):
        super(noflayer, self).__init__()
        self.alpha = 0.2
        self.variant = variant
        self.nnode = nnode
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        #self.adj = adj
        self.adj = adj.to_dense()
        #self.adj = adj.to_dense() - torch.diag(torch.diag(adj.to_dense())) 
        self.sign_mask = torch.zeros((adj.shape[0], self.in_features)).cuda()
        
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.a = nn.Parameter(torch.empty(size=(2*self.in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.relu = nn.ReLU()
        #self.f = Parameter(torch.ones(self.nnode))
        self.weight_matrix_att = Parameter(torch.FloatTensor(self.in_features, self.in_features))
        #self.weight_matrix2 = Parameter(torch.FloatTensor(2*self.in_features, self.in_features))
        self.temp = Parameter(torch.Tensor(5))
        self.weight_matrix_att_prime_1 = torch.nn.Parameter(torch.Tensor(self.in_features, self.in_features))
        #self.weight_matrix_att_prime_2 = torch.nn.Parameter(torch.Tensor(2*self.in_features, 1))
      #self.weight_matrix_att_prime_3 = torch.nn.Parameter(torch.Tensor(2*self.in_features, 1))
        self.rowsum = torch.ones((self.adj.shape[0])).cuda()
        
        self.reset_parameters()

    def reset_parameters(self):
        #torch.nn.init.xavier_uniform_(self.weight_matrix_att)
        #torch.nn.init.xavier_uniform_(self.weight)
        #torch.nn.init.xavier_uniform_(self.weight_matrix2)
        self.temp.data.fill_(0.0)
        #torch.nn.init.xavier_uniform_(self.weight_matrix_att_prime_1)
        #torch.nn.init.xavier_uniform_(self.weight_matrix_att_prime_2)
        #torch.nn.init.xavier_uniform_(self.weight_matrix_att_prime_3)
        #torch.nn.init.xavier_uniform_(self.f)
        #torch.nn.init.xavier_uniform_(self.a)
        #stdv = 1. / math.sqrt(self.out_features)
        #self.weight.data.uniform_(-stdv, stdv)
        #self.weight_matrix_att.data.uniform_(-stdv, stdv)
        #self.weight_matrix.data.uniform_(-stdv, stdv)

    def attention(self, feature):
        #feature = torch.mm(feature, self.weight_matrix_att)
        feat_1 = torch.matmul(feature, self.a[:self.in_features, :])
        feat_2 = torch.matmul(feature, self.a[self.in_features:, :])
        
        #sam_1 = torch.matmul(feature, self.b[:self.in_features, :].clone())
        #sam_2 = torch.matmul(feature, self.b[self.in_features:, :].clone())
        
        # broadcast add
        e = feat_1 + feat_2.T
        e = self.leakyrelu(e)
        
        #s = sam_1 + sam_2.T
        #s = self.leakyrelu(s)
        
        zero_vec = -9e15*torch.ones_like(e)
        att = torch.where(self.adj > 0, e, zero_vec)
        att = F.softmax(att, dim=1)
        
        # raf_2 = att.multinomial(num_samples=self.max_degree, replacement=True)
        # self.zero[self.ind, raf_2] = 1
        # zero = torch.where(self.zero > 0, e, zero_vec)
        # zero = F.softmax(zero, dim=1)
        
        
        
        # att_s = torch.where(self.adj > 0, s, zero_vec)
        # att_s = F.softmax(att_s, dim=1).clone()
        
        # for i in range(len(self.raf_indx)):
        #     index = att[self.raf_indx[i],:].multinomial(num_samples=self.raf_len[i], replacement=False)
        #     att[self.raf_indx[i],index] = 0
        
        
        #raf_1 = att.multinomial(num_samples=35, replacement=False)
        # tensor_raf_indx = torch.tensor(self.raf_indx)
        # tensor_raf_indx = tensor_raf_indx.reshape((tensor_raf_indx.shape[0],1))
        # raf_1 = att[self.raf_indx].multinomial(num_samples=35, replacement=False).detach().cpu()
        # raf_att = torch.where(att[tensor_raf_indx,raf_1] !=0, att[tensor_raf_indx,raf_1],0)
        
        U = att
        P = 0.5*U
        #adj_nonzero_ind = att.nonzero(as_tuple=True)
        return U,P
    
    def forward_lifting_bases(self, feature, prop_u, P, U, h0, rowsum, layer, coe):
        
        #predict = torch.mm(self.adj, feature)
        
        #prop_u = torch.mm(U, feature)
        
        #AP = torch.mul(self.adj, P)
        #rowsum = torch.sum(AP,1)
        xi = torch.einsum('ij,i->ij', feature.float(), rowsum.float())
        feat_odd_bar = prop_u - xi
        
        #einsum = torch.einsum('ij,i->ij', feature.float(), self.rowsum.float())
        #feat_odd_bar = predict - einsum
        
        #update = torch.mm(U, feature)
        #UP = torch.mul(U, P)
        #rowsum = torch.sum(UP,1)
        #feat_even_bar = feature + update - torch.einsum('ij,i->ij', feature.float(), rowsum.float())
        feat_even_bar = feature + prop_u - xi
        #feat_even_bar = feature + update - einsum
        
        #feat_odd_bar = coe[0]*feat_odd_bar + (1-coe[0])*h0
        #feat_odd_bar = coe[0]*feat_odd_bar + (1-coe[0])*h0
        feat = coe[2]*feat_odd_bar + (1-coe[2])*feat_even_bar
        return feat
    
    def soft_thresholding(self, feat, threshold):
        sign_mask = self.sign_mask
        sign_mask[feat < 0] = -1
        sign_mask[feat > 0] = 1
        mod_feat = torch.abs(feat)
        mod_feat[mod_feat > threshold] -= threshold
        mod_feat[mod_feat <= threshold] = 0
        #mod_feat = torch.where(mod_feat>threshold, 0.5*mod_feat, 0)
        feat = torch.mul(mod_feat.clone(), sign_mask.clone())
        return feat
    
    def inverse_lifting_bases(self, feat, P, U, layer, rowsum, h0, coe):
        #update = torch.mm(U, feat)
        prop_u = torch.mm(U, feat)
        feat_odd_bar = feat - coe[3]*prop_u
        
        #predict = torch.mm(self.adj, feat)
        #AP = torch.mul(self.adj, P)
        #rowsum = torch.sum(AP,1)
        
        #feat_even_bar = predict + torch.einsum('ij,i->ij', feat_odd_bar.float(), rowsum.float())
        feat_even_bar = prop_u + torch.einsum('ij,i->ij', feat_odd_bar.float(), rowsum.float())
        
        feat_odd_bar = coe[0]*feat_odd_bar + (1-coe[0])*h0
        feat_prime = coe[1]*feat_even_bar + (1-coe[1])*feat_odd_bar
        #feat_prime = coe[0]*feat_prime + (1-coe[0])*h0
        #feat_prime = 1*feat_even_bar + 0.0*feat_odd_bar
        
        return feat_prime

    
        
    def forward(self, input, h0, l):
        #beta = math.log(lamda/l+1)
        hi = input
        U,P = self.attention(hi)
        prop_u = torch.mm(U, hi)
        UP = torch.mul(U, P)
        rowsum = torch.sum(UP,1)
        coe = torch.sigmoid(self.temp)
        feat = self.forward_lifting_bases(hi, prop_u, P, U, h0, rowsum, l, coe)
        
        feat = self.soft_thresholding(feat, threshold=10**-6)
        
        output = self.inverse_lifting_bases(feat, P, U, l, rowsum, h0, coe)
        return output