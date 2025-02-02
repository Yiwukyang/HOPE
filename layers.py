from typing import Optional, Callable
import manifolds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add

from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros

import manifolds

class ProposedConv(MessagePassing):#超图卷积层
    _cached_norm_n2e: Optional[Tensor]
    _cached_norm_e2n: Optional[Tensor]

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0, #隐藏层维度：节点到边的维度，输出层维度：边到节点的维度
                 act: Callable = nn.PReLU(), bias: bool = True, cached: bool = False,
                 row_norm: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')#额外参数：聚合方式
        kwargs.setdefault('flow', 'source_to_target')#额外参数：流动方向
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm

        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot')
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')#特征空间映射（点到边，边到点）

        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias_n2e', None) 
            self.register_parameter('bias_e2n', None) 
        
        self.reset_parameters()#所有参数初始化
    
    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        self._cached_norm_n2e = None
        self._cached_norm_e2n = None
    
    def forward(self, x: Tensor, hyperedge_index: Tensor, 
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):

        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:#如果未显式提供节点和超边数量，则从输入数据推断
            num_edges = int(hyperedge_index[1].max()) + 1

        cache_norm_n2e = self._cached_norm_n2e
        cache_norm_e2n = self._cached_norm_e2n

        if (cache_norm_n2e is None) or (cache_norm_e2n is None):
            hyperedge_weight = x.new_ones(num_edges)

            node_idx, edge_idx = hyperedge_index
            Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                             hyperedge_index[0], dim=0, dim_size=num_nodes)
            De = scatter_add(x.new_ones(hyperedge_index.shape[1]),
                             hyperedge_index[1], dim=0, dim_size=num_edges)#计算节点度数和超边度数

            if self.row_norm:#计算节点到超边和超边到节点的归一化因子，举个例子：一条超边连接三个节点，那么这条超边对每个节点的贡献都是三分之一，反之亦然
                Dn_inv = 1.0 / Dn
                Dn_inv[Dn_inv == float('inf')] = 0
                De_inv = 1.0 / De
                De_inv[De_inv == float('inf')] = 0

                norm_n2e = De_inv[edge_idx]
                norm_e2n = Dn_inv[node_idx]
                
            else:
                Dn_inv_sqrt = Dn.pow(-0.5)
                Dn_inv_sqrt[Dn_inv_sqrt == float('inf')] = 0
                De_inv_sqrt = De.pow(-0.5)
                De_inv_sqrt[De_inv_sqrt == float('inf')] = 0

                norm = De_inv_sqrt[edge_idx] * Dn_inv_sqrt[node_idx]
                norm_n2e = norm
                norm_e2n = norm

            if self.cached:#归一化因子保存
                self._cached_norm_n2e = norm_n2e
                self._cached_norm_e2n = norm_e2n
        else:
            norm_n2e = cache_norm_n2e
            norm_e2n = cache_norm_e2n

        x = self.lin_n2e(x)#特征线性变换
        e = self.propagate(hyperedge_index, x=x, norm=norm_n2e, #特征聚合，利用最下面那个函数，将节点的特征和归一因子相乘得到边的特征
                               size=(num_nodes, num_edges))  # Node to edge

        if self.bias_n2e is not None:
            e = e + self.bias_n2e
        e = self.act(e)
        e = F.dropout(e, p=self.dropout, training=self.training)

        x = self.lin_e2n(e)
        n = self.propagate(hyperedge_index.flip([0]), x=x, norm=norm_e2n, #使用翻转的索引（边到节点）聚合特征
                               size=(num_edges, num_nodes))  # Edge to node

        if self.bias_e2n is not None:
            n = n + self.bias_e2n

        #n = self.act(n)
        #n = F.dropout(n, p=self.dropout, training=self.training)
        #n = self.act(n)
        return n, e # No act, act

    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j


class Hyperbolid_ProposedConv(MessagePassing):  # 超图卷积层
    _cached_norm_n2e: Optional[Tensor]
    _cached_norm_e2n: Optional[Tensor]

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, args, dropout: float = 0.0,  # 隐藏层维度：节点到边的维度，输出层维度：边到节点的维度
                 act: Callable = nn.PReLU(), bias: bool = True, cached: bool = False,
                 row_norm: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')  # 额外参数：聚合方式
        kwargs.setdefault('flow', 'source_to_target')  # 额外参数：流动方向
        super().__init__(node_dim=0, **kwargs)
        self.manifold = getattr(manifolds, args.manifold)()
        self.c_hypergraph = torch.tensor([args.c_hypergraph]).to(args.device)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm

        self.lin_n2e = Linear(in_dim + 1, hid_dim, bias=False, weight_initializer='glorot')
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')  # 特征空间映射（点到边，边到点）

        self.proj_head = nn.Sequential(Linear(self.hid_dim, self.hid_dim), ReLU(inplace=True),
                                       Linear(self.hid_dim, self.hid_dim))

        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias_n2e', None)
            self.register_parameter('bias_e2n', None)

        self.reset_parameters()  # 所有参数初始化

    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        self._cached_norm_n2e = None
        self._cached_norm_e2n = None

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):

        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:  # 如果未显式提供节点和超边数量，则从输入数据推断
            num_edges = int(hyperedge_index[1].max()) + 1

        cache_norm_n2e = self._cached_norm_n2e
        cache_norm_e2n = self._cached_norm_e2n

        if (cache_norm_n2e is None) or (cache_norm_e2n is None):
            hyperedge_weight = x.new_ones(num_edges)

            node_idx, edge_idx = hyperedge_index
            Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                             hyperedge_index[0], dim=0, dim_size=num_nodes)
            De = scatter_add(x.new_ones(hyperedge_index.shape[1]),
                             hyperedge_index[1], dim=0, dim_size=num_edges)  # 计算节点度数和超边度数

            if self.row_norm:  # 计算节点到超边和超边到节点的归一化因子，举个例子：一条超边连接三个节点，那么这条超边对每个节点的贡献都是三分之一，反之亦然
                Dn_inv = 1.0 / Dn
                Dn_inv[Dn_inv == float('inf')] = 0
                De_inv = 1.0 / De
                De_inv[De_inv == float('inf')] = 0

                norm_n2e = De_inv[edge_idx]
                norm_e2n = Dn_inv[node_idx]

            else:
                Dn_inv_sqrt = Dn.pow(-0.5)
                Dn_inv_sqrt[Dn_inv_sqrt == float('inf')] = 0
                De_inv_sqrt = De.pow(-0.5)
                De_inv_sqrt[De_inv_sqrt == float('inf')] = 0

                norm = De_inv_sqrt[edge_idx] * Dn_inv_sqrt[node_idx]
                norm_n2e = norm
                norm_e2n = norm

            if self.cached:  # 归一化因子保存
                self._cached_norm_n2e = norm_n2e
                self._cached_norm_e2n = norm_e2n
        else:
            norm_n2e = cache_norm_n2e
            norm_e2n = cache_norm_e2n

        o = torch.zeros_like(x)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.expmap0(x, self.c_hypergraph)
        x = self.manifold.proj(x, c=self.c_hypergraph)
        x = self.manifold.logmap0(x, c=self.c_hypergraph)

        x = self.lin_n2e(x)  # 特征线性变换

        e = self.propagate(hyperedge_index, x=x, norm=norm_n2e,  # 特征聚合，利用最下面那个函数，将节点的特征和归一因子相乘得到边的特征
                           size=(num_nodes, num_edges))  # Node to edge

        if self.bias_n2e is not None:
            e = e + self.bias_n2e
        e = self.act(e)
        e = F.dropout(e, p=self.dropout, training=self.training)

        x = self.lin_e2n(e)

        n = self.propagate(hyperedge_index.flip([0]), x=x, norm=norm_e2n,  # 使用翻转的索引（边到节点）聚合特征
                           size=(num_edges, num_nodes))  # Edge to node

        if self.bias_e2n is not None:
            n = n + self.bias_e2n

        n = self.act(n)
        n = F.dropout(n, p=self.dropout, training=self.training)

        #n = self.proj_head(n)
        #e = self.proj_head(e)

        n = self.manifold.expmap0(n, c=self.c_hypergraph)
        n = self.manifold.proj(n, c=self.c_hypergraph)
        n = self.manifold.logmap0(n, c=self.c_hypergraph)
        #n = self.manifold.proj(n, c=self.c_hypergraph)

        e = self.manifold.expmap0(e, c=self.c_hypergraph)
        e = self.manifold.proj(e, c=self.c_hypergraph)
        e = self.manifold.logmap0(e, c=self.c_hypergraph)
        #e = self.manifold.proj(e, c=self.c_hypergraph)

        # n = self.act(n)
        # n = F.dropout(n, p=self.dropout, training=self.training)
        # n = self.act(n)
        return n, e  # No act, act

    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j


