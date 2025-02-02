from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add

import manifolds

from layers import ProposedConv, Hyperbolid_ProposedConv
from torch_geometric.nn import GCNConv
from torch.nn import Sequential, Linear, ReLU


class hyperbolid_HyperEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, num_layers=2, act: Callable = nn.PReLU(), args = None):
        super(hyperbolid_HyperEncoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = act
        self.args = args
        self.convs = nn.ModuleList()  # convs是一个模块列表，包含多个proposedconv层
        if num_layers == 1:
            self.convs.append(Hyperbolid_ProposedConv(self.in_dim, self.edge_dim, self.node_dim, args=self.args, cached=False, act=act))
        else:
            self.convs.append(Hyperbolid_ProposedConv(self.in_dim, self.edge_dim, self.node_dim, args=self.args, cached=False, act=act))
            for _ in range(self.num_layers - 2):
                self.convs.append(Hyperbolid_ProposedConv(self.node_dim, self.edge_dim, self.node_dim, args=self.args, cached=False, act=act))
            self.convs.append(Hyperbolid_ProposedConv(self.node_dim, self.edge_dim, self.node_dim, args=self.args, cached=False, act=act))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int):
        for i in range(self.num_layers):  # 按顺序执行每一行前传
            x, e = self.convs[i](x, hyperedge_index, num_nodes, num_edges)  # 对于每一层，输入节点特征，超边索引，输出更新后节点特征和超边特征
            x = self.act(x)  # 每层卷积后引入非线性激活函数
        return x, e  # act, act


class HyperEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, num_layers=2, act: Callable = nn.PReLU()):
        super(HyperEncoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = act
        self.convs = nn.ModuleList()#convs是一个模块列表，包含多个proposedconv层
        if num_layers == 1:
            self.convs.append(ProposedConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        else:
            self.convs.append(ProposedConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            for _ in range(self.num_layers - 2):
                self.convs.append(ProposedConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            self.convs.append(ProposedConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))

        
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int):
        for i in range(self.num_layers):#按顺序执行每一行前传
            x, e = self.convs[i](x, hyperedge_index, num_nodes, num_edges)#对于每一层，输入节点特征，超边索引，输出更新后节点特征和超边特征
            x = self.act(x)#每层卷积后引入非线性激活函数
        return x, e # act, act



class LightHGCL(nn.Module):
    def __init__(self, encoder: HyperEncoder, proj_dim: int, args):
        super(LightHGCL, self).__init__()
        self.encoder = encoder

        self.manifold = getattr(manifolds, args.manifold)()
        self.c = torch.tensor([args.c]).to(args.device)

        self.in_dim = self.encoder.in_dim
        self.node_dim = self.encoder.node_dim
        self.edge_dim = self.encoder.edge_dim

        self.fc1_n = nn.Linear(self.node_dim, proj_dim)
        self.fc2_n = nn.Linear(proj_dim, self.node_dim)
        self.fc1_e = nn.Linear(self.edge_dim, proj_dim)
        self.fc2_e = nn.Linear(proj_dim, self.edge_dim)

        self.proj_head = nn.Sequential(Linear(proj_dim, proj_dim), ReLU(inplace=True),
                                       Linear(proj_dim, proj_dim))

        #self.edge_dim = self.encoder.edge_dim

        # self.proj_1 = nn.Sequential(
        #   nn.Linear(self.node_dim, proj_dim),
        #   nn.Batchnorm1d(proj_dim),
        #   nn.ReLU(),
        #   nn.Linear(proj_dim, 64)
        # )
        # self.proj_2 = nn.Sequential(
        #   nn.Linear(self.node_dim, proj_dim),
        #   nn.Batchnorm1d(proj_dim),
        #   nn.ReLU(),
        #   nn.Linear(proj_dim, 64)
        # )
#         self.fc1_n = nn.Linear(self.node_dim, proj_dim)
#         self.fc2_n = nn.Linear(proj_dim, self.node_dim)
        #self.fc1_e = nn.Linear(self.edge_dim, proj_dim)
        #self.fc2_e = nn.Linear(proj_dim, self.edge_dim)
        self.act = nn.ReLU()

        #self.disc = nn.Bilinear(self.node_dim, self.edge_dim, 1)
        self.conv1 = GCNConv(self.in_dim, self.node_dim)
        self.conv2 = GCNConv(self.node_dim, self.node_dim)

        self.conv3 = GCNConv(self.in_dim + 1, self.node_dim)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
#         self.fc1_n.reset_parameters()
#         self.fc2_n.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.fc1_n.reset_parameters()
        self.fc2_n.reset_parameters()
        self.fc1_e.reset_parameters()
        self.fc2_e.reset_parameters()
        #self.fc1_e.reset_parameters()
        #self.fc2_e.reset_parameters()
        #self.disc.reset_parameters()
        
    def forward(self, x: Tensor, hyperedge_index: Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1#若没有点数和边数，则从超边索引中推断

        node_idx = torch.arange(0, num_nodes, device=x.device)
        edge_idx = torch.arange(num_edges, num_edges + num_nodes, device=x.device)
        self_loop = torch.stack([node_idx, edge_idx])#构造自环
        self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
        n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + num_nodes)#超图卷积获得点边特征
        return n, e[:num_edges]


    def forward_gcn(self, x: Tensor, edge_index: Tensor, edge_attr = None):#使用两层 GCN 对输入特征 x 进行卷积
        x = self.conv1(x, edge_index, edge_attr)
        x = self.act(x)
        x = F.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        #x = self.conv2(x, edge_index, edge_attr)
        return x


    def hyperbolid_forward_gcn(self, x: Tensor, edge_index: Tensor):
        o = torch.zeros_like(x)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.expmap0(x, self.c)
        x = self.manifold.proj(x, c=self.c)
        x = self.manifold.logmap0(x, c=self.c)
        x = self.conv3(x, edge_index)
        x = self.act(x)
        x = F.dropout(x)
        x = self.conv2(x, edge_index)
        #x = self.proj_head(x)
        x = self.manifold.expmap0(x, c=self.c)
        x = self.manifold.proj(x, c=self.c)
        #x = self.manifold.logmap0(x, c=self.c)
        #x = self.manifold.proj(x, c=self.c)
        return x


    def without_selfloop(self, x: Tensor, hyperedge_index: Tensor, node_mask: Optional[Tensor] = None,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1

        if node_mask is not None:
            node_idx = torch.where(~node_mask)[0]
            edge_idx = torch.arange(num_edges, num_edges + len(node_idx), device=x.device)
            self_loop = torch.stack([node_idx, edge_idx])
            self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
            n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + len(node_idx))
            return n, e[:num_edges]
        else:
            return self.encoder(x, hyperedge_index, num_nodes, num_edges)

    def f(self, x, tau):
        return torch.exp(x / tau)

    def node_projection(self, z: Tensor):
        return self.fc2_n(F.elu(self.fc1_n(z)))

    def edge_projection(self, z: Tensor):
        return self.fc2_e(F.elu(self.fc1_e(z)))
    
    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # def disc_similarity(self, z1: Tensor, z2: Tensor):
    #     return torch.sigmoid(self.disc(z1, z2)).squeeze()

    def __semi_loss(self, h1: Tensor, h2: Tensor, tau: float, num_negs: Optional[int]):
        if num_negs is None:
#             refl_sim = self.f(self.cosine_similarity(h1, h1), tau)
#             between_sim = self.f(self.cosine_similarity(h1, h2), tau)

#             return -torch.log(
#                 between_sim.diag()
#                 / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

            #h2 = self.manifold.expmap0(h2, c=self.c)
            #h2 = self.manifold.proj(h2, c=self.c)

            #h1 = self.manifold.logmap0(h1, c=self.c)
            #h1 = self.manifold.proj(h1, c=self.c)

            #between_sim = self.f(self.manifold.dist(h1, h2, c = self.c), tau)
            between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            return -torch.log(between_sim.diag() / between_sim.sum(1))
        else:
            pos_sim = self.f(F.cosine_similarity(h1, h2), tau)
            #pos_sim = self.f(self.manifold.dist(h1, h2, c = self.c), tau)
            negs = []
            for _ in range(num_negs):
                negs.append(h2[torch.randperm(h2.size(0))])
            #negs = torch.stack(negs, dim=-1)
            neg_sim = self.f(F.cosine_similarity(h1.unsqueeze(-1).tile(num_negs), negs), tau)
            #neg_sim = self.f(self.manifold.dist(h1.unsqueeze(-1).tile(num_negs), negs), tau)
            return -torch.log(pos_sim / (pos_sim + neg_sim.sum(1)))
        
    def __semi_loss_batch(self, h1: Tensor, h2: Tensor, tau: float, batch_size: int):
        device = h1.device
        num_samples = h1.size(0)
        num_batches = (num_samples - 1) // batch_size + 1
        indices = torch.arange(0, num_samples, device=device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]

            #h2 = self.manifold.expmap0(h2, c=self.c)
            #h2 = self.manifold.proj(h2, c=self.c)

            #h1 = self.manifold.logmap0(h1, c=self.c)
            #h1 = self.manifold.proj(h1, c=self.c)

            between_sim = self.f(self.cosine_similarity(h1[mask], h2), tau)
            #between_sim = self.f(self.manifold.dist(h1[mask], h2, c = self.c), tau)
            loss = -torch.log(between_sim[:, i * batch_size: (i + 1) * batch_size].diag() / between_sim.sum(1))
            losses.append(loss)
        return torch.cat(losses)

    def __loss(self, z1: Tensor, z2: Tensor, tau: float, batch_size: Optional[int], #组合正向和反向的对比损失，并根据需要进行批量化计算。
               num_negs: Optional[int], mean: bool):
        if batch_size is None or num_negs is not None:
            l1 = self.__semi_loss(z1, z2, tau, num_negs)
            l2 = self.__semi_loss(z2, z1, tau, num_negs)
        else:
            l1 = self.__semi_loss_batch(z1, z2, tau, batch_size)
            l2 = self.__semi_loss_batch(z2, z1, tau, batch_size)

        loss = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()
        return loss

    def node_level_loss(self, n1: Tensor, n2: Tensor, node_tau: float, 
                       batch_size: Optional[int] = None, num_negs: Optional[int] = None, 
                       mean: bool = True):
        loss = self.__loss(n1, n2, node_tau, batch_size, num_negs, mean)
        return loss
    
    def group_level_loss(self, e1: Tensor, e2: Tensor, edge_tau: float, 
                       batch_size: Optional[int] = None, num_negs: Optional[int] = None, 
                       mean: bool = True):
        loss = self.__loss(e1, e2, edge_tau, batch_size, num_negs, mean)
        return loss
