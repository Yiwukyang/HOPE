from collections import defaultdict
from typing import Optional
import os.path as osp
import pickle

import torch
import numpy as np
from scipy import sparse
from torch_scatter import scatter_add
from torch.utils.data import random_split
from utils import clique_expansion, clique_expansion_weight, hyperedge_clique_expansion


class BaseDataset(object):
    def __init__(self, type: str, name: str, device: str = 'cpu'):
        self.type = type
        self.name = name
        self.device = device
        self.path = './dataset'
        if self.type in ['cocitation', 'coauthorship']:
            self.dataset_dir = osp.join(self.path, self.type, self.name) 
        else:
            self.dataset_dir = osp.join(self.path, self.name)
        self.split_dir = osp.join(self.dataset_dir, 'splits')

        self.load_dataset()
        self.preprocess_dataset()

    def load_dataset(self):
        with open(osp.join(self.dataset_dir, 'features.pickle'), 'rb') as f:
            self.features = pickle.load(f)
        with open(osp.join(self.dataset_dir, 'hypergraph.pickle'), 'rb') as f:
            self.hypergraph = pickle.load(f)
        with open(osp.join(self.dataset_dir, 'labels.pickle'), 'rb') as f:
            self.labels = pickle.load(f)

    def load_splits(self, seed: int):
        with open(osp.join(self.split_dir, f'{seed}.pickle'), 'rb') as f:
            splits = pickle.load(f)
        return splits
        
    def preprocess_dataset(self):
        edge_set = set(self.hypergraph.keys())
        edge_to_num = {}
        num_to_edge = {}
#         max_value = max([max(each) for each in list(self.hypergraph.values())])

#         weak_label = [0 for i in range(max_value+1)]
        num = 0
        for edge in edge_set:
            edge_to_num[edge] = num
            num_to_edge[num] = edge
            num += 1

        incidence_matrix = []
        processed_hypergraph = {}#一个字典，将超边编号映射到与其关联的节点集合中
        for edge in edge_set:#遍历超边
            nodes = self.hypergraph[edge]#获取超边对应的节点集合
            processed_hypergraph[edge_to_num[edge]] = nodes
            #edge_to_num[edge] 将超边 edge 转换为一个唯一的编号（数字）。processed_hypergraph 字典中记录了这个编号对应的节点集合 nodes。
            for node in nodes:
                incidence_matrix.append([node, edge_to_num[edge]])
                #weak_label[node] = edge_to_num[edge]

        node_to_edges = {}
        for node, edge in incidence_matrix:
            if node not in node_to_edges:
                node_to_edges[node] = []
            node_to_edges[node].append(edge)

        self.node_to_edges = node_to_edges
        self.incidence_matrix = incidence_matrix
        self.processed_hypergraph = processed_hypergraph
        #self.features = torch.as_tensor(self.features)#(self.features.toarray())
        self.features = torch.as_tensor(self.features.toarray())
        self.hyperedge_index = torch.LongTensor(incidence_matrix).T.contiguous()#关联矩阵转置，第一行是节点索引，第二行是边索引
        #self.adjacency_index, self.edge_attr = clique_expansion_weight(self.hyperedge_index)
        self.adjacency_index = clique_expansion(self.hyperedge_index)
        self.hyperedge_adjacency_index = hyperedge_clique_expansion(self.hyperedge_index)
        self.labels = torch.LongTensor(self.labels)
        self.num_nodes = int(self.hyperedge_index[0].max()) + 1
        self.num_edges = int(self.hyperedge_index[1].max()) + 1
        self.edge_to_num = edge_to_num
        self.num_to_edge = num_to_edge
        #self.weak_label = torch.LongTensor(weak_label)
        self.to(self.device)

    def to(self, device: str):
        self.features = self.features.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.adjacency_index = self.adjacency_index.to(device)
        self.hyperedge_adjacency_index = self.hyperedge_adjacency_index.to(device)
        #self.edge_attr = self.edge_attr.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self

    def generate_random_split(self, train_ratio: float = 0.1, val_ratio: float = 0.1,
                              seed: Optional[int] = None, use_stored_split: bool = True):
        if use_stored_split:
            splits = self.load_splits(seed)
            train_mask = torch.tensor(splits['train_mask'], dtype=torch.bool, device=self.device)
            val_mask = torch.tensor(splits['val_mask'], dtype=torch.bool, device=self.device)
            test_mask = torch.tensor(splits['test_mask'], dtype=torch.bool, device=self.device)

        else:
            num_train = int(self.num_nodes * train_ratio)
            num_val = int(self.num_nodes * val_ratio)
            num_test = self.num_nodes - (num_train + num_val)

            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
            else:
                generator = torch.default_generator

            train_set, val_set, test_set = random_split(
                torch.arange(0, self.num_nodes), (num_train, num_val, num_test), 
                generator=generator)
            train_idx, val_idx, test_idx = \
                train_set.indices, val_set.indices, test_set.indices
            train_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
            val_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
            test_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

        return [train_mask, val_mask, test_mask]



class CoraCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cocitation', 'cora', **kwargs)


class CiteseerCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cocitation', 'citeseer', **kwargs)


class PubmedCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cocitation', 'pubmed', **kwargs)


class CoraCoauthorshipDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('coauthorship', 'cora', **kwargs)


class DBLPCoauthorshipDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('coauthorship', 'dblp', **kwargs)


class ZooDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', 'zoo', **kwargs)


class NewsDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', '20newsW100', **kwargs)


class MushroomDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', 'Mushroom', **kwargs)


class NTU2012Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'NTU2012', **kwargs)


class ModelNet40Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'ModelNet40', **kwargs)


class DBLPCopubDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', 'dblp_copub', **kwargs)


class AminerDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', 'aminer', **kwargs)


class HouseDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', 'house', **kwargs)


class IMDBDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', 'imdb', **kwargs)