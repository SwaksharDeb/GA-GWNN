from typing import Tuple, Optional, Union
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul_
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.dense.linear import Linear

from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul_(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul_(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class GCNIIdenseConv(MessagePassing):
    
    _cached_edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = True,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        super(GCNIIdenseConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        heads = 1
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        #self.add_self_loops = add_self_loops
        
        self.add_self_loops = False
        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(torch.Tensor(in_channels, out_channels))
        self.temp = Parameter(torch.Tensor(4))
        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.weight_matrix_att = Parameter(torch.FloatTensor(in_channels, out_channels))
        self.a = nn.Parameter(torch.empty(size=(2*in_channels, 1)))
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        self.lin_src = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
        self.lin_dst = self.lin_src
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # if bias == 'bn':
        #     self.norm = nn.BatchNorm1d(out_channels)
        # elif bias == 'ln':
        #     self.norm = nn.LayerNorm(out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        # if bias:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)


        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None
        glorot(self.att_src)
        glorot(self.att_dst)
        self.temp.data.fill_(0.0)
        torch.nn.init.xavier_uniform_(self.weight_matrix_att)
        #zeros(self.bias)
        
    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=0.2, training=self.training)
        return alpha

    
    def forward(self, x: Tensor, edge_index: Adj, alpha, h0,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        H = 1
        C =  self.out_channels
        coe = torch.sigmoid(self.temp)
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        #support = x + torch.matmul(x, self.weight1)
        support = torch.matmul(x, self.weight1)
        AP = torch.mul(edge_weight, 0.5*edge_weight)
        sparse_tensor = torch.sparse_coo_tensor(edge_index, AP, size=(x.size(self.node_dim), x.size(self.node_dim)))
        rowsum = torch.sparse.sum(sparse_tensor,1)
        rowsum = rowsum.to_dense()
        
        predict = self.propagate(edge_index, x=support, edge_weight=edge_weight, size=None)
        feat_odd_bar = predict - torch.einsum('ij,i->ij', support.float(), rowsum.float())
        update = self.propagate(edge_index, x=support, edge_weight=edge_weight, size=None)
        feat_even_bar = support + update - torch.einsum('ij,i->ij', support.float(), rowsum.float())
        feat = coe[0]*feat_odd_bar + (1-coe[0])*feat_even_bar
        
        """filtering"""
        #feat = self.soft_thresholding(feat, threshold=10**-6)
        
        """Inverse lifting"""
        update = self.propagate(edge_index, x=feat, edge_weight=edge_weight, size=None)
        feat_odd_bar = feat - coe[1]*update
        predict = self.propagate(edge_index, x=feat, edge_weight=edge_weight, size=None)
        feat_even_bar = predict + torch.einsum('ij,i->ij', feat_odd_bar.float(), rowsum.float())
        #feat_odd_bar = coe[2]*feat_odd_bar + (1-coe[2])*h0
        out = coe[2]*feat_even_bar + (1-coe[2])*feat_odd_bar
        #feat_prime = coe[0]*feat_prime + (1-coe[0])*out
        
        out = self.norm(out)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        assert edge_weight is not None
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)