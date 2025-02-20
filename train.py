import argparse
import os
import random
import yaml
from tqdm import tqdm
import numpy as np
import torch
from sklearn.manifold import TSNE
from loader import DatasetLoader
from sklearn.cluster import KMeans
from models import HyperEncoder, LightHGCL, hyperbolid_HyperEncoder
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from utils import drop_features, drop_incidence, valid_node_edge_mask, hyperedge_index_masking, clique_expansion, \
    hyperedge_clique_expansion
from evaluation import linear_evaluation, linear_evaluation_other
from optimizers.radam import RiemannianAdam

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(model_type, num_negs):
    features, hyperedge_index, adjacency_index, processed_hypergraph, hyperedge_adjacency_index, incidence_matrix= data.features, data.hyperedge_index, \
    data.adjacency_index, data.processed_hypergraph, data.hyperedge_adjacency_index, data.incidence_matrix
    num_nodes, num_edges = data.num_nodes, data.num_edges#data中变量提取
    #weak_label = data.weak_label
    #edge_attr = data.edge_attr

    model.train()
    optimizer.zero_grad()

    # Hypergraph Augmentation
    # hyperedge_index1 = drop_incidence(hyperedge_index, params['drop_incidence_rate'])
    # hyperedge_index2 = drop_incidence(hyperedge_index, params['drop_incidence_rate'])
    # x1 = drop_features(features, params['drop_feature_rate'])
    # x2 = drop_features(features, params['drop_feature_rate'])

    # node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
    # node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
    # node_mask = node_mask1 & node_mask2
    # edge_mask = edge_mask1 & edge_mask2

    # Hypergraph Expansion
    # adjacency_matrix = clique_expansion(hyperedge_index)

    # Encoder模型前传
    n1, e = model(features, hyperedge_index, num_nodes, num_edges)#生成超图节点和超边表示
    e1 = [torch.mean(features[index], dim=0, keepdim=True) for index in processed_hypergraph.values()]
    e1 = torch.cat(e1, dim=0)  # 所有超边嵌入拼接成一个张量[1579,1433]
    n2 = model.forward_gcn(features, adjacency_index)#计算普通图的节点表示
    n3 = model.forward_gcn(e1, hyperedge_adjacency_index)

    #n2 = model.hyperbolid_forward_gcn(features, adjacency_index)  # 计算普通图的节点表示
    #n3 = model.hyperbolid_forward_gcn(e1, hyperedge_adjacency_index)


    #e2 = [torch.mean(n2[index], dim=0, keepdim=True) for index in processed_hypergraph.values()]
    #e2 = torch.cat(e2, dim=0)



    # 计算新的点表示
    #new_node_features = [torch.mean(n3[index], dim=0, keepdim=True) for index in data.node_to_edges.values()]
    #new_node_features = torch.cat(new_node_features, dim=0)
    #e = [torch.sum(n1[index], dim=0, keepdim=True) for index in processed_hypergraph.values()]
    #e = torch.cat(e, dim=0)
    #普通节点表示生成超边表示

#     indices = torch.LongTensor([index for index in processed_hypergraph.values()])
#     gathered_n2 = torch.gather(n2, 0, indices.view(-1, 1).expand(-1, n2.size(1)))
#     e2 = torch.mean(gathered_n2, dim=0, keepdim=True)
    #print(e2.size())

    #print(e2.size())
    #print(n3.size())
    # n1, e1 = model(x1, hyperedge_index1, num_nodes, num_edges)
    # n2, e2 = model(x2, hyperedge_index2, num_nodes, num_edges)

    loss_n = model.node_level_loss(n1 , n2 , params['tau_n'], batch_size=params['batch_size_2'], num_negs=num_negs)
    #loss_h = model.node_level_loss(n1, new_node_features, params['tau_n'], batch_size=params['batch_size_2'], num_negs=num_negs)
    #loss_g = model.group_level_loss(e + n3, e2 , params['tau_g'], batch_size=params['batch_size_1'], num_negs=num_negs)
    loss_g = model.group_level_loss(e, n3, params['tau_g'], batch_size=params['batch_size_2'], num_negs=num_negs)
    #loss_e = model.group_level_loss(e, e2, params['tau_g'], batch_size=params['batch_size_2'], num_negs=num_negs)

    loss = loss_n + params['wg'] * loss_g # +  loss_e
    loss.backward()
    optimizer.step()
    return loss.item()


def node_classification_eval(num_splits=20, mode='test'):
    model.eval()
    n, e = model(data.features, data.hyperedge_index)
    #n = model.hyperbolid_forward_gcn(data.features, data.hyperedge_index)
    #n_g = model.forward_gcn(data.features, data.adjacency_index)
    #e1 = [torch.sum(data.features[index], dim=0, keepdim=True) for index in data.processed_hypergraph.values()]
    #e1 = torch.cat(e1, dim=0)
    #n3 = model.forward_gcn(e1, data.hyperedge_adjacency_index)
    #n_e = [torch.mean(n3[index], dim=0, keepdim=True) for index in data.node_to_edges.values()]
    #n_e = torch.cat(n_e, dim=0)

    #n = torch.cat([n, n1], dim=-1)

    if data.name == 'pubmed':
        lr = 0.005
        max_epoch = 300
    elif data.name == 'cora' or data.name == 'citeseer':
        lr = 0.005
        max_epoch = 300
    elif data.name == 'Mushroom':
        lr = 0.01
        max_epoch = 200
    else:
        lr = 0.01
        max_epoch = 100

    accs = []
    for i in range(num_splits):
        masks = data.generate_random_split(seed=i)
        accs.append(linear_evaluation(n, data.labels, masks, lr=lr, max_epoch=max_epoch, mode=mode))
        #accs.append(linear_evaluation_other(n, data.labels, masks, lr=lr, max_epoch=max_epoch, mode=mode, dataset=data.name))
    
    return accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HOPE Contrastive Learning.')
    parser.add_argument('--dataset', type=str, default='pubmed',
        choices=['cora', 'citeseer', 'pubmed', 'dblp_coauthor', 'house', 'imdb', 'zoo', 'NTU2012', 'ModelNet40', 'dblp_copub', 'aminer'])
    parser.add_argument('--model_type', type=str, default='HOPE')
    #parser.add_argument('--wg', type=float, default=10)
    #parser.add_argument('--tau_n', type=float, default=0.5)
    #parser.add_argument('--tau_g', type=float, default=0.58)
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--eval', type=str, default='cla')
    parser.add_argument('--device', type=int, default=8)
    #parser.add_argument('--optimizer', type=str, default='Adam',help='which optimizer to use, can be any of [Adam, RiemannianAdam]')
    #parser.add_argument('--momentum', type=float, default=0.95, help='momentum in optimizer')
    #parser.add_argument('--pooling', type=str, default='add', choices=['add', 'max', 'mean'])
    #parser.add_argument('--readout', type=str, default='concat', choices=['concat', 'add', 'last'])
    parser.add_argument('--manifold', type=str, default='PoincareBall', choices=['PoincareBall','Hyperboloid'])
    parser.add_argument('--c', type=float, default=0)
    parser.add_argument('--c_hypergraph', type=float, default=1)

    args = parser.parse_args()

    params = yaml.safe_load(open('config.yaml'))[args.dataset]
    #params['tau_n'], params['tau_g'] = args.tau_n, args.tau_g
    #params['w_g'] = args.wg
    print(params)

    data = DatasetLoader().load(args.dataset).to(args.device)

    path = './savepoint/'
    #path_cluster = './new_savecluster/'
    accs = []
    for seed in range(args.num_seeds):
        fix_seed(42)
        best_valid_acc = 0
        best_valid_nmi = 0

        #encoder = HyperEncoder(data.features.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'])
        encoder = hyperbolid_HyperEncoder(data.features.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'], args = args)
        model = LightHGCL(encoder, params['proj_dim'], args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        #optimizer = RiemannianAdam(params=model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'], momentum=0.95)

        if args.eval == 'cla':#节点分类任务
            for epoch in tqdm(range(1, params['epochs'] + 1)):
                loss = train(args.model_type, num_negs=None)#每次损失值，参数：模型类型，是否使用负样本
                if (epoch + 1) % 50 == 0:#每50轮执行一次验证
                    valid_acc = np.mean(node_classification_eval(mode='valid'))#平均准确率

                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        print(f'\n epoch: {epoch}, valid_acc: {valid_acc:.3f}')
                        torch.save(model.state_dict(), path+args.dataset+'_model.pkl')#当前模型参数保存路径

        model.load_state_dict(torch.load(path+args.dataset+'_model.pkl'))
        
        if args.eval == 'cla':
            acc = node_classification_eval()

            accs.append(acc)
            acc_mean, acc_std = np.mean(acc, axis=0), np.std(acc, axis=0)
            #print(accs)
            print(f'seed: {seed}, test_acc: {acc_mean:.2f}+-{acc_std:.2f}')

#     accs = np.array(accs).reshape(-1, 3)

#     accs = np.array(accs)
#     accs_mean = np.mean(accs)
#     accs_std = np.std(accs)
#     print(f'[Final] dataset: {args.dataset}, test_acc: {accs_mean:.2f}+-{accs_std:.2f}')