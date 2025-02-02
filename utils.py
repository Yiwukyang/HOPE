import random
from itertools import permutations, combinations

import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def drop_features(x: Tensor, p: float):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]


def drop_incidence(hyperedge_index: Tensor, p: float = 0.2):
    if p == 0.0:
        return hyperedge_index
    
    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p
    
    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)
    return hyperedge_index


def drop_nodes(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_nodes, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[drop_idx, :] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def drop_hyperedges(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_edges, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[:, drop_idx] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def valid_node_edge_mask(hyperedge_index: Tensor, num_nodes: int, num_edges: int):
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1])
    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def common_node_edge_mask(hyperedge_indexs, num_nodes, num_edges):
    hyperedge_weight = hyperedge_indexs[0].new_ones(num_edges)
    node_mask = hyperedge_indexs[0].new_ones((num_nodes,)).to(torch.bool)
    edge_mask = hyperedge_indexs[0].new_ones((num_edges,)).to(torch.bool)

    for index in hyperedge_indexs:
        Dn = scatter_add(hyperedge_weight[index[1]], index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(index.new_ones(index.shape[1]), index[1], dim=0, dim_size=num_edges)
        node_mask &= Dn != 0
        edge_mask &= De != 0
    return node_mask, edge_mask


def hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask, edge_mask):
    if node_mask is None and edge_mask is None:
        return hyperedge_index

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = H[node_mask].to_sparse().indices()
    return masked_hyperedge_index


def clique_expansion_weight(edge_index):
    edge_weight_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = np.sort(edge_index[0, :][edge_index[1, :] == he])
        if len(nodes_in_he) == 1:
            continue  # skip self loops
        combs = combinations(nodes_in_he, 2)
        for comb in combs:
            if not comb in edge_weight_dict.keys():
                edge_weight_dict[comb] = 1 / len(nodes_in_he)
            else:
                edge_weight_dict[comb] += 1 / len(nodes_in_he)
                
    new_edge_index = np.zeros((2, len(edge_weight_dict)))
    new_norm = np.zeros((len(edge_weight_dict)))
    cur_idx = 0
    for edge in edge_weight_dict:
        new_edge_index[:, cur_idx] = edge
        new_norm[cur_idx] = edge_weight_dict[edge]
        cur_idx += 1

    edge_index = torch.tensor(new_edge_index).type(torch.LongTensor)
    norm = torch.tensor(new_norm).type(torch.FloatTensor)
    return edge_index, norm


def clique_expansion(hyperedge_index: Tensor):
    edge_set = set(hyperedge_index[1].tolist())#获取所有超边索引，set去重
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        #对于每个超边 (edge)，我们首先根据超边索引 hyperedge_index[1] == edge 创建一个掩码 mask，该掩码用于选择属于该超边的所有节点。
        #然后，我们提取该超边的节点列表（nodes）
        for e in permutations(nodes, 2):#对这些节点，使用 itertools.permutations(nodes, 2) 生成所有可能的 2-节点对，表示超边中的每一对节点。permutations(nodes, 2) 会生成节点的所有 2 元组
            adjacency_matrix.append(e)

    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    #将去重后的节点对列表转换为 torch.LongTensor，并转置 .T 使得最终的形状为 [2, num_edges]（每列是一个节点对，第一行为节点1，第二行为节点2）
    return adjacency_matrix.to(hyperedge_index.device)
    #最后，将生成的邻接矩阵移动到与 hyperedge_index 相同的设备（GPU或CPU），以确保张量的设备一致性。

#new
'''
def hyperedge_clique_expansion(hyperedge_index: Tensor):
    node_set = set(hyperedge_index[0].tolist())  # 获取所有点索引，set去重
    hyperedge_adjacency_matrix = []
    for node in node_set:
        mask = hyperedge_index[0] == node
        hyperedges = hyperedge_index[:, mask][1].tolist()
        # 对于每个点，我们首先根据点索引 hyperedge_index[1] == edge 创建一个掩码 mask，该掩码用于选择属于该点的所有超边。
        # 然后，我们提取该点的超边列表（hyperedges）
        for e in permutations(hyperedges, 2):  # 对这些节点，使用 itertools.permutations(nodes, 2) 生成所有可能的 2-节点对，表示超边中的每一对节点。permutations(nodes, 2) 会生成节点的所有 2 元组
            hyperedge_adjacency_matrix.append(e)

    hyperedge_adjacency_matrix = list(set(hyperedge_adjacency_matrix))
    hyperedge_adjacency_matrix = torch.LongTensor(hyperedge_adjacency_matrix).T.contiguous()
    # 将去重后的超边对列表转换为 torch.LongTensor，并转置 .T （每列是一个超边对，第一行为超边1，第二行为超边2）
    return hyperedge_adjacency_matrix.to(hyperedge_index.device)

'''

def hyperedge_clique_expansion(hyperedge_index: Tensor, threshold: float = 0):
    """
    对超图进行超边展开，确保邻接矩阵对称，且包含自连接。
    """
    edge_set = set(hyperedge_index[1].tolist())  # 获取所有超边索引
    processed_hypergraph = {}  # 保存每条超边的节点集合
    for edge in edge_set:
        mask = hyperedge_index[1] == edge  # 选择属于该超边的所有节点
        nodes = set(hyperedge_index[0][mask].tolist())
        processed_hypergraph[edge] = nodes  # 保存超边及其关联的节点集合

    hyperedge_adjacency_matrix = []  # 用于保存符合条件的超边对
    for edge1, nodes1 in processed_hypergraph.items():
        for edge2, nodes2 in processed_hypergraph.items():
            # 计算两个超边的节点交集大小及比例
            intersection_size = len(nodes1 & nodes2)
            if (
                intersection_size / len(nodes1) > threshold  # 超过超边1的阈值
                and intersection_size / len(nodes2) > threshold  # 超过超边2的阈值
            ):
                hyperedge_adjacency_matrix.append((edge1, edge2))
                if edge1 != edge2:  # 仅当 edge1 ≠ edge2 时，添加对称项
                    hyperedge_adjacency_matrix.append((edge2, edge1))

    # 转换为张量格式并返回
    hyperedge_adjacency_matrix = torch.LongTensor(hyperedge_adjacency_matrix).T.contiguous()
    return hyperedge_adjacency_matrix.to(hyperedge_index.device)


'''
def hyperedge_clique_expansion(
        hyperedge_index: Tensor,
        start: float = 0.0,
        end: float = 1.0,
        step: float = 0.05
):
    """
    对超图进行超边展开并生成不同阈值下的邻接矩阵，确保邻接矩阵对称，且包含自连接。

    Args:
        hyperedge_index (Tensor): 超图的超边索引。
        start (float): 阈值的起始值，默认 0.0。
        end (float): 阈值的结束值，默认 1.0。
        step (float): 阈值的递增步长，默认 0.05。

    Returns:
        dict: 每个阈值对应的邻接矩阵，字典格式 {threshold: adjacency_matrix}。
    """
    edge_set = set(hyperedge_index[1].tolist())  # 获取所有超边索引
    processed_hypergraph = {}  # 保存每条超边的节点集合
    for edge in edge_set:
        mask = hyperedge_index[1] == edge  # 选择属于该超边的所有节点
        nodes = set(hyperedge_index[0][mask].tolist())
        processed_hypergraph[edge] = nodes  # 保存超边及其关联的节点集合

    adjacency_matrices = {}  # 保存不同阈值的邻接矩阵

    # 遍历不同的阈值
    threshold = start
    while threshold <= end:
        print(f"Generating adjacency matrix for threshold = {threshold:.2f}")

        hyperedge_adjacency_matrix = []  # 用于保存符合条件的超边对

        for edge1, nodes1 in processed_hypergraph.items():
            for edge2, nodes2 in processed_hypergraph.items():
                # 计算两个超边的节点交集大小及比例
                intersection_size = len(nodes1 & nodes2)
                if (
                        intersection_size / len(nodes1) > threshold  # 超过超边1的阈值
                        and intersection_size / len(nodes2) > threshold  # 超过超边2的阈值
                ):
                    hyperedge_adjacency_matrix.append((edge1, edge2))
                    if edge1 != edge2:  # 仅当 edge1 ≠ edge2 时，添加对称项
                        hyperedge_adjacency_matrix.append((edge2, edge1))

        # 转换为张量格式
        adjacency_matrix = torch.LongTensor(hyperedge_adjacency_matrix).T.contiguous()
        adjacency_matrices[threshold] = adjacency_matrix.to(hyperedge_index.device)

        # 增加阈值
        threshold += step

    return adjacency_matrices
    '''
