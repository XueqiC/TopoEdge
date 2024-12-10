import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_self_loops, degree
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from torch_scatter import scatter_add, scatter_mean
from collections import defaultdict
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from collections import Counter
import math

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# warnings.filterwarnings('ignore', category=UserWarning, message=".*Sparse CSR tensor support is in beta state.*")


def batch_to_gpu(batch, device):
    for c in batch:
        batch[c] = batch[c].to(device)

    return batch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# def cal_reweight(train_y):
#     reweight = torch.tensor([(train_y == _).sum().item() for _ in torch.unique(train_y)])
#     reweight = 1/reweight
#     reweight /= reweight.sum()

#     return reweight
#     # return torch.tensor([0.0001,1])

def cal_reweight(train_y, num_classes):
    # Initialize a tensor with zeros for each class
    reweight = torch.zeros(num_classes)
    for cls in torch.unique(train_y):
        reweight[cls] = (train_y == cls).sum().item()
    reweight = reweight.sum() / reweight
    reweight[reweight == float('inf')] = reweight[reweight != float('inf')].max()
    return reweight
    
def process(edge_index, edge_attr):
    edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes = edge_index.max().item() + 1, reduce = 'mean')
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes = edge_index.max().item() + 1, fill_value="mean")
    return edge_index, edge_attr

def process_edge(edge_index):
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes = edge_index.max().item() + 1)

    return edge_index

def process_edge_attr(edge_index, edge_feature):
    edge_index, edge_feature = to_undirected(edge_index, edge_feature, num_nodes = edge_index.max().item() + 1, reduce = 'mean')
    #transform to long type of edge_attr
    
    return edge_index, edge_feature.type(torch.LongTensor)

def mean_encoding(edge_labels, edge_features):
    # Given edge_labels and edge_features (here it means label diffusion encoding) 
    # Get universe pattrn encoding and std for each class
    num_labels = edge_labels.max().item() + 1
    num_features = edge_features.size(1)
    expanded_labels = edge_labels.unsqueeze(-1).expand(-1, num_features)
    summed_features = torch.zeros(num_labels, num_features, device=edge_features.device)
    summed_features.scatter_add_(0, expanded_labels, edge_features)
    count = torch.bincount(edge_labels, minlength=num_labels).float().clamp(min=1).unsqueeze(-1)
    mean_matrix = summed_features / count
    
    squared_diff = (edge_features - mean_matrix[edge_labels])**2
    variance = torch.zeros(num_labels, num_features, device=edge_features.device)
    variance.scatter_add_(0, expanded_labels, squared_diff) / count
    std_matrix = torch.sqrt(variance.clamp(min=1e-8))  # add a small value to avoid division by zero
    
    return mean_matrix, std_matrix

def normalize_encoding(edge_labels, edge_features, mean_matrix, std_matrix):
    # Get deviation encoding
    normalized_edge_encoding = ((edge_features - mean_matrix[edge_labels])**2) / std_matrix[edge_labels]
    # normalized_edge_encoding = ((edge_features - mean_matrix[edge_labels])**2) 
    # print(normalized_edge_encoding.shape, std_matrix.shape, )
    # l2_norm = torch.sqrt(normalized_edge_encoding.sum(dim=1))
    # return l2_norm
    return normalized_edge_encoding

def norm(edge_features):
    # Get deviatio score
    # Compute the squared differences
    squared_diff = (edge_features)**2
    # Take the L2 norm (i.e., the square root of the sum of squared differences across each row)
    l2_norm = torch.sqrt(squared_diff.sum(dim=1))
    
    return l2_norm

def heatmap(mean_matrix, args):
    mean_matrix_np = mean_matrix.numpy()
    plt.figure(figsize=(10, 8))
    plt.rcParams["font.family"] = "Times New Roman"
    sns.heatmap(mean_matrix_np, annot=True, cmap="YlGnBu")
    plt.xlabel('Label Class')
    plt.ylabel('Label Class')
    plt.title('Mean values of diffusion edge label encoding for each class')
    if args.task == 0:
        plt.savefig('./image/Heatmap/' + args.dataset + '_' + args.method + '_' + str(args.mixup) + '.png', bbox_inches='tight')
        plt.close()
    elif args.task == 1:
        plt.savefig('./image/Heatmap/' + args.dataset + '_' + args.method + '_' + str(args.mixup) + '_multi.png', bbox_inches='tight')
        plt.close()

def histogram(edge_label, score, args):
    edge_label_np = edge_label.numpy()
    score_np = score.numpy()

    # Get unique labels
    unique_labels = np.unique(edge_label_np)

    # Plot histograms for each edge_label
    for label in unique_labels:
        plt.figure()  # Start a new figure for each label
        plt.hist(score_np[edge_label_np == label], bins=100, alpha=0.7, color='blue')
        plt.xlabel('Deviation Weight')
        plt.ylabel('Count')
        plt.title('Edge Label '+ str(label))
        
        if args.task == 0:
            plt.savefig('./image/histogram/' + args.dataset + '_' + str(label)+ 'new.png', bbox_inches='tight')
            plt.close()
        elif args.task == 1:
            plt.savefig('./image/histogram/' + args.dataset + '_' + str(label)+  'new_multi.png', bbox_inches='tight')
            plt.close()
        
def mean_degrees(edges, node_degrees):
    source_degrees= node_degrees[edges[0]]
    target_degrees= node_degrees[edges[1]]
    return source_degrees.mean().item(), target_degrees.mean().item()

def n_degrees(edge_index):
    # Calculate the degree of each node and ensure it is of long datatype
    node_degrees = degree(edge_index[0], num_nodes=edge_index.max().item() + 1, dtype=torch.long) + \
                degree(edge_index[1], num_nodes=edge_index.max().item() + 1, dtype=torch.long)

    # Prepare to store neighborhood degrees for each edge
    neighborhood_degrees = []

    for edge in edge_index.t():
        i, j = edge.tolist()

        # Find neighbors of node i while excluding direct edges to node j
        neighbors_of_i = ((edge_index[0] == i) & (edge_index[1] != j)) | ((edge_index[1] == i) & (edge_index[0] != j))
        # exclude_j = (edge_index[0] != j) & (edge_index[1] != j)
        # neighbors_of_i = neighbors_of_i & exclude_j

        # Get degrees of these neighbors
        neighbors_nodes = edge_index[0][neighbors_of_i] + edge_index[1][neighbors_of_i] - i
        degrees = node_degrees[neighbors_nodes]
        
        neighborhood_degrees.append(degrees)
    return neighborhood_degrees

def one_sim(edge_index, edge_attr):
    num_nodes = edge_index.max().item() + 1
    num_classes = edge_attr.max().item() + 1
    num_edges = edge_index.size(1)
    
    deg = degree(edge_index[0], num_nodes=num_nodes)
    deg1 = degree(edge_index[1], num_nodes=num_nodes)
    d = deg + deg1
    i = d[edge_index[0]]
    j = d[edge_index[1]]

    a=torch.bincount(edge_attr, minlength=num_classes)/edge_attr.size(0)
    encoding_i = a * (i.unsqueeze(1) - 1 ) / i.unsqueeze(1) 
    encoding_j = a * (j.unsqueeze(1) - 1 ) / j.unsqueeze(1) 
    encoding_i[torch.arange(edge_attr.size(0)), edge_attr] += 1 / i
    encoding_j[torch.arange(edge_attr.size(0)), edge_attr] += 1 / j
    
    n_degree1 = n_degrees(edge_index)
    swapped_edge_index = edge_index.clone()
    swapped_edge_index[[0, 1]] = swapped_edge_index[[1, 0]]
    n_degree2 = n_degrees(swapped_edge_index)
    
    encoding_li = torch.zeros(num_edges, num_classes)
    encoding_lj = torch.zeros(num_edges, num_classes)
    for n in range(num_edges):
        n_degree1[n] = n_degree1[n].unsqueeze(1)
        n_degree2[n] = n_degree2[n].unsqueeze(1)
    # W/o Self-loop
    #     encoding_li[n,:] = a * (i[n] - 1).view(1,-1) - a * (1/n_degree1[n]).sum().view(1,-1) + encoding_j[n].view(1,-1) * ((1/n_degree1[n]).sum()).view(1,-1) 
    #     encoding_lj[n,:] = a * (j[n] - 1).view(1,-1) - a * (1/n_degree2[n]).sum().view(1,-1) + encoding_i[n].view(1,-1) * ((1/n_degree2[n]).sum()).view(1,-1)
    # encoding_ri = (a * (j.unsqueeze(1) - 1) + encoding_i)/ j.unsqueeze(1)
    # encoding_i0 = (encoding_li+encoding_ri)/i.unsqueeze(1)
    # encoding_rj = (a * (i.unsqueeze(1) - 1) + encoding_j)/ i.unsqueeze(1)
    # encoding_j0 = (encoding_lj+encoding_rj)/j.unsqueeze(1)
    
    # w/ Self-loop
        encoding_li[n,:] = a * (i[n] - 1).view(1,-1) - a * (1/(n_degree1[n]+1)).sum().view(1,-1) + encoding_j[n].view(1,-1) * ((1/(n_degree1[n]+1)).sum()).view(1,-1) 
        encoding_lj[n,:] = a * (j[n] - 1).view(1,-1) - a * (1/(n_degree2[n]+1)).sum().view(1,-1) + encoding_i[n].view(1,-1) * ((1/(n_degree2[n]+1)).sum()).view(1,-1)
    # print(n_degree1[n].shape)
    encoding_ri = (a * (j.unsqueeze(1) - 1) + encoding_i + encoding_j)/ (j.unsqueeze(1) + 1)
    encoding_i0 = (encoding_li + encoding_ri + encoding_i)/(i.unsqueeze(1) + 1)
    encoding_rj = (a * (i.unsqueeze(1) - 1) + encoding_j + encoding_i)/ (i.unsqueeze(1) + 1)
    encoding_j0 = (encoding_lj + encoding_rj + encoding_j)/(j.unsqueeze(1) + 1)
    
    encoding = (encoding_i0+encoding_j0)/2
    fig, axes = plt.subplots(num_classes, num_classes, figsize=(5*num_classes, 5*num_classes))
    for i in range(num_classes):  # Iterate over rows/classes
        for j in range(num_classes):  # Iterate over columns/dimensions
            # Select the encoding of the edges for class i
            class_mask = edge_attr == i
            class_encoding = encoding[class_mask, j]

            # Calculate the mean
            mean_val = class_encoding.mean().item()
            ax = axes[i, j]
            ax.hist(class_encoding.numpy(), bins=40, color='indigo', alpha=0.7, range=(0, 1))
            ax.set_title(f'Class {i} - Dim {j} - Mean: {mean_val:.4f}', fontsize = 20)
            ax.set_xlabel('Value', fontsize=16)  # Set X-axis label font size
            ax.set_ylabel('Frequency', fontsize=20) 
            ax.tick_params(axis='x', labelsize=20)  # Set x-axis tick label size for this particular subplot
            ax.tick_params(axis='y', labelsize=20)  # Set x-axis tick label size for this particular subplot

    # Adjust the layout
    plt.tight_layout()
    plt.savefig('./image/density/derviation_binary.png', dpi = 200, transparent = True, bbox_inches = 'tight')
    plt.show()
    
    # fig, axes = plt.subplots(4, 1, figsize=(10, 20))
    # for i in range(4):  # Assume there are at least 4 classes
    #     class_mask = edge_attr == i
    #     class_encoding = encoding[class_mask]
        
    #     # Plotting each dimension as a line in the distribution
    #     for dim in range(num_classes):
    #         sns.kdeplot(class_encoding[:, dim].numpy(), ax=axes[i], label=f'Dim {dim}', clap = True)
        
    #     axes[i].set_title(f'Class {i} Encoding Distribution')
    #     axes[i].legend()

    # plt.tight_layout()
    # plt.savefig('./image/density/all_values_0000.png')
    # plt.show()

    # pos_i, pos_j = i[edge_attr == 0], j[edge_attr == 0]
    # neg_i, neg_j = i[edge_attr == 1], j[edge_attr == 1]
    # a = edge_attr[edge_attr == 0].shape[0]/edge_attr.shape[0]
    # b = edge_attr[edge_attr == 1].shape[0]/edge_attr.shape[0]
    # pos_i0, pos_i1 = (a*(pos_i-1)+1)/pos_i, b*(pos_i-1)/pos_i
    # pos_j0, pos_j1 = (a*(pos_j-1)+1)/pos_j, b*(pos_j-1)/pos_j
    # neg_i0, neg_i1 = a*(neg_i-1)/neg_i, (b*(neg_i-1)+1)/neg_i
    # neg_j0, neg_j1 = a*(neg_j-1)/neg_j, (b*(neg_j-1)+1)/neg_j
    # # all_i0, all_i1 = (a*(i-1))/i, (b*(i-1)+1)/i
    # # all_j0, all_j1 = (a*(j-1))/j, (b*(j-1)+1)/j
    # # all_i0, all_i1 = ((i+1)*a*(i-1)+1)/(i**2), ((i+1)*b*(i-1))/(i**2)
    # # all_j0, all_j1 = ((j+1)*a*(j-1)+1)/(j**2), ((j+1)*b*(j-1))/(j**2)
    # all_i0, all_i1 = ((i+1)*a*(i-1))/(i**2), ((i+1)*b*(i-1)+1)/(i**2)
    # all_j0, all_j1 = ((j+1)*a*(j-1))/(j**2), ((j+1)*b*(j-1)+1)/(j**2)

    # plt.figure(figsize=(2*num_classes, 2*num_classes))
    # plt.rcParams["font.family"] = "Times New Roman"
    # data_i = [pos_i0, pos_i1, neg_i0, neg_i1]
    # titles = ['Class 0, Dim 0', 'Class 0, Dim 1', 'Class 1, Dim 0', 'Class 1, Dim 1']

    # for k in range(4):
    #     plt.subplot(2, 2, k+1)
    #     sns.histplot(data_i[k], kde=False, bins=10)
    #     plt.title(f'{titles[k]}, mean = {data_i[k].mean():.4f}')
    #     plt.xlabel('Value')
    #     plt.ylabel('Density')

    # plt.tight_layout()
    # plt.savefig('./image/density/i1_values.png')
    # plt.show()
    
    # data_i = [all_i0, all_i1, all_j0, all_j1]
    # titles = ['i_node, Dim 0', 'i_node, Dim 1', 'j_node, Dim 0', 'j_node, Dim 1']

    # for k in range(4):
    #     plt.subplot(2, 2, k+1)
    #     sns.histplot(data_i[k], kde=False, bins=10)
    #     plt.title(f'{titles[k]}, mean = {data_i[k].mean():.4f}')
    #     plt.xlabel('Value')
    #     plt.ylabel('Density')

    # plt.tight_layout()
    # # plt.savefig('./image/density/alln_values.png')
    # plt.savefig('./image/ap/alln_values.png')
    # plt.show()

    # Plot for 'j' values
    # plt.figure(figsize=(12, 12))
    # data_j = [pos_j0, pos_j1, neg_j0, neg_j1]

    # for k in range(4):
    #     plt.subplot(2, 2, k+1)
    #     sns.histplot(data_j[k], kde=False, bins=10)
    #     plt.title(f'{titles[k]}, mean = {data_j[k].mean():.4f}')
    #     plt.xlabel('Value')
    #     plt.ylabel('Density')

    # plt.tight_layout()
    # plt.savefig('./image/density/j1_values.png')
    # plt.show()
    
    
def cal_topo_reweight(edge_index, edge_attr, args):
# The code `args.reweight = cal_reweight(train_y)` calculates the reweighting factor for each
# class in the training data. It uses the `cal_reweight` function to compute the reweighting
# factor based on the class distribution. The reweighting factor is a measure of class
# imbalance, where classes with fewer samples are assigned higher weights to balance the
# training process.

# The above code is calculating the local homophily ratio for a given set of edges and their
# attributes. It takes in the edge index, edge attributes, and two arguments 'n' and 'w'. It then
# calculates the ratios and assigns them to the variable 'ratios'. Finally, it assigns the value of
# 'ratios' to the variable 'weight'.
    # ratios = local_homo_ratio(edge_index, edge_attr, args.n, args.w)
    # weight = ratios
    # d1 = degree(edge_index[0])
    # d2 = degree(edge_index[1])
    # d3 = (d1[edge_index[0]]+d2[edge_index[1]])/2
    # weight = d3
    
    edge_index2, edge_attr2 = process_edge_attr(edge_index, edge_attr)
    num_classes = torch.unique(edge_attr2).shape[0]
    max_edge_attr = edge_attr2.max().item()
    edge_one_hot = torch.zeros(edge_attr2.shape[0], max(max_edge_attr + 1, num_classes))
    # edge_one_hot = torch.zeros(edge_attr2.shape[0], torch.unique(edge_attr2).shape[0])
    edge_one_hot[torch.arange(edge_attr2.shape[0]), edge_attr2] = 1
    node_one_hot = scatter_add(edge_one_hot, edge_index2[0], dim = 0)/degree(edge_index2[0]).view(-1, 1)
    
    num_classes = edge_attr.max() + 1  # Assuming edge_attr contains class labels starting from 0
    # edge_update = []
    # for cls in range(num_classes):
    #     cls_mask = edge_attr == cls
    #     indices0 = edge_index[0, cls_mask]
    #     indices1 = edge_index[1, cls_mask]
    #     edge_update.append((node_one_hot[indices0]+node_one_hot[indices1])/2)

    edge_index2, _ = add_self_loops(edge_index2, num_nodes = edge_index2.max().item() + 1)
    node_deg = degree(edge_index2[0])
    deg_inv_sqrt = 1/torch.sqrt(node_deg)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    # edge_weight = deg_inv_sqrt[edge_index2[0]] * deg_inv_sqrt[edge_index2[1]]
    edge_weight = deg_inv_sqrt[edge_index2[0]] * deg_inv_sqrt[edge_index2[0]]
    adj_t = torch.sparse_coo_tensor(edge_index2, edge_weight, torch.Size([node_deg.shape[0], node_deg.shape[0]]))
    adj_power = adj_t
    
    # Method 1: A^K
    for _ in range(1, args.n):
        adj_power = torch.sparse.mm(adj_power, adj_t)
    diff_node_one_hot = torch.sparse.mm(adj_power, node_one_hot)
    
    # # Method 2: 1/k*(A+A^2+...+A^K)
    # adj_sum = adj_t.clone()
    # for _ in range(1, args.n):
    #     adj_power = torch.sparse.mm(adj_power, adj_t)
    #     adj_sum += adj_power
    # adj_avg = adj_sum / args.n
    # diff_node_one_hot = torch.sparse.mm(adj_avg, node_one_hot)
    
    diff_edge_one_hot = (diff_node_one_hot[edge_index[0]] + diff_node_one_hot[edge_index[1]])/2
    edge_update = []
    for cls in range(num_classes):
        cls_mask = edge_attr == cls
        class_features = diff_edge_one_hot[cls_mask]
        # class_features = diff_edge_one_hot
        edge_update.append(class_features)
        # indices0 = edge_index[0, cls_mask]
        # indices1 = edge_index[1, cls_mask]
        # edge_update.append((diff_node_one_hot [indices0]+diff_node_one_hot[indices1])/2)
    
    entropy2 = Categorical(probs = diff_edge_one_hot).entropy()
    unique_attrs, inverse_indices = torch.unique(edge_attr, return_inverse=True)
    # mean_encoding = scatter_mean(diff_edge_one_hot, inverse_indices, dim=0)
    mean_encoding = scatter_mean(diff_edge_one_hot, inverse_indices, dim=0)
    # weight = entropy2 
    
    # ====================
    # a=torch.tensor([(edge_attr2 == _).sum().item() for _ in torch.unique(edge_attr2)])
    # a=a/(a.sum())
    # # diff_edge_one_hot = diff_edge_one_hot - a
    # node_degrees = scatter_add(torch.ones(edge_index2.size(1)), edge_index2[0], dim=0)
    # d = node_degrees.mean().item()
    # a0=1/(d**2)*torch.tensor([(d+1)*a[0]*(d-1)+1, (d+1)*a[1]*(d-1)]).reshape(-1,2)
    # a1=1/(d**2)*torch.tensor([(d+1)*a[0]*(d-1), (d+1)*a[1]*(d-1)+1]).reshape(-1,2)
    # edge_attr = edge_attr.reshape(-1,1)
    # diff_edge_one_hot = torch.where(edge_attr == 0, diff_edge_one_hot - a0, diff_edge_one_hot - a1)
    # weight = torch.norm(diff_edge_one_hot, p=2, dim=1)
    # ========================================
    
    # # diff_edge_one_hot is the label diffusion encoding
    # mean_matrix, std_matrix = mean_encoding(edge_attr, diff_edge_one_hot)
    # norm_matrix = normalize_encoding(edge_attr, diff_edge_one_hot, mean_matrix, std_matrix)
    # entropy1 = norm(norm_matrix)
    # entropy1 = norm((diff_edge_one_hot-mean_matrix[edge_attr]))
    # # Get deviation weight
    # weight = 1 / (entropy1+0.01)
    # # weight = entropy1 + 1
    
    # mean_class_edge_dist = scatter_mean(diff_edge_one_hot, edge_attr, dim = 0)
    # deviation = diff_edge_one_hot - mean_class_edge_dist[edge_attr]
    # norm = (mean_class_edge_dist**2).sum(dim = 1)**0.5

    # deviation = ((deviation**2).sum(dim = 1)**0.5/norm[edge_attr])**2
    # weight = 1/deviation

    # histogram(edge_attr, weight, args)
    
    # entropy2 = Categorical(probs = diff_edge_one_hot).entropy()
    # uni_class = torch.unique(edge_attr)
    # num_class = [(edge_attr == _).sum().item() for _ in uni_class]
    # max_class = max(num_class)
    # w = max_class/torch.tensor(num_class)
    # weight = entropy2 
    
    
    # heatmap(mean_matrix, args)
    
    # selected_values = diff_edge_one_hot[torch.arange(diff_edge_one_hot.size(0)), edge_attr]
    # row_sums = diff_edge_one_hot.sum(dim=1)
    # rest_sums = row_sums - selected_values
    # rest_averages = rest_sums / (diff_edge_one_hot.size(1) - 1)
    # # result = selected_values - rest_averages
    # result = rest_averages - selected_values 
    # # min_value = -1/((len(uni_class))-1)
    # # max_value = 1
    # max_value = 1/((len(uni_class))-1)
    # min_value = -1
    # weight = ((result - min_value) / (max_value - min_value)) + 0.5

    # ave_entropy2 = torch.tensor([entropy2[edge_attr == _].mean().item() for _ in uni_class])

    # print(entropy2.shape, ave_entropy2.shape, edge_attr.shape, w.shape, edge_attr.shape)
    # weight = max(entropy2) - entropy2 +0.5
    # weight = entropy2
    
    # weight = (entropy2 - ave_entropy2[edge_attr])/w[edge_attr]
    
    # weight = (weight - weight.min() + 1e-2)/(weight.max() - weight.min() + 1e-2)
    # print(weight.shape, weight.min(), weight.max())

    return edge_update
    # return mean_encoding, ratio_i_pos, ratio_j_pos, ratio_i_neg, ratio_j_neg
    # return weight
    # return diff_node_one_hot[edge_index[0]], diff_node_one_hot[edge_index[1]], entropy2

def el_ratio(edge_index, edge_attr):
    num_nodes = int(edge_index.max()) + 1
    counts = torch.zeros((num_nodes, 2))
    
    node1, node2 = edge_index
    label_index = torch.stack([node1, node2, node2, node1]).long()
    label_value = torch.stack([edge_attr, edge_attr, edge_attr, edge_attr])
    counts.index_add_(0, label_index.view(-1), torch.nn.functional.one_hot(label_value, num_classes=2).view(-1, 2).float())
    ratios = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
    edge_head_ratios = ratios[edge_index[0]][:,0]
    edge_tail_ratios = ratios[edge_index[1]][:,0]
    edge_ratios = torch.cat((edge_head_ratios.unsqueeze(1), edge_tail_ratios.unsqueeze(1)), dim=1)
    edge_ratios = edge_ratios.view(-1, 2)
    
    return edge_ratios

def quan_post(ratios, thre):
    labels = torch.zeros(ratios.shape[0], dtype=torch.long)
    
    # Define conditions using the provided abbreviations
    MM = (ratios[:, 0] > thre) & (ratios[:, 1] > thre)
    MU = ((ratios[:, 0] > thre) & ((ratios[:, 1] > (1-thre)) & (ratios[:, 1] < thre))) | \
        ((ratios[:, 1] > thre) & ((ratios[:, 0] > (1-thre)) & (ratios[:, 0] < thre)))
    Mm = ((ratios[:, 0] > thre) & (ratios[:, 1] < (1-thre))) | \
        ((ratios[:, 1] > thre) & (ratios[:, 0] < (1-thre)))
    UU = ((ratios[:, 0] > (1-thre)) & (ratios[:, 0] < thre)) & \
        ((ratios[:, 1] > (1-thre)) & (ratios[:, 1] < thre))
    Um = ((ratios[:, 0] > (1-thre)) & (ratios[:, 0] < thre) & (ratios[:, 1] < (1-thre))) | \
        ((ratios[:, 1] > (1-thre)) & (ratios[:, 1] < thre) & (ratios[:, 0] < (1-thre)))
    mm = (ratios[:, 0] < (1 - thre)) & (ratios[:, 1] < (1 - thre))
    labels[MM] = 0
    labels[MU] = 1
    labels[Mm] = 2
    labels[UU] = 3
    labels[Um] = 4
    labels[mm] = 5
    mean_values = ratios.mean(dim=1)
    return mean_values, labels

def ge_post(enc1, enc2, thre):
    enp1 = Categorical(probs = enc1).entropy()
    enp2 = Categorical(probs = enc2).entropy()
    
    labels = []
    threshold = thre  # Define the threshold value

    for e1, e2 in zip(enc1, enc2):
        # RR
        if e1[0] > threshold and e2[0] > threshold:
            labels.append(0)
        # RU
        elif (e1[0] > threshold and max(e2[0], e2[1]) <= threshold) or (e2[0] > threshold and max(e1[0], e1[1]) <= threshold):
            labels.append(1)
        # RB
        elif (e1[0] > threshold and e2[1] > threshold) or (e2[0] > threshold and e1[1] > threshold):
            labels.append(2)
        # UU
        elif max(e1[0], e1[1]) <= threshold and max(e2[0], e2[1]) <= threshold:
            labels.append(3)
        # UB
        elif (max(e1[0], e1[1]) <= threshold and e2[1] > threshold) or (max(e2[0], e2[1]) <= threshold and e1[1] > threshold):
            labels.append(4)
        # BB
        elif e1[1] > threshold and e2[1] > threshold:
            labels.append(5)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return enp1, enp2, labels_tensor

def topo_imb_ratio(data):
    entropy=cal_topo_reweight(data.edge_index, data.y.view(-1)).cpu()
    k= torch.unique(data.y).shape[0]
    H=torch.zeros((1,k))
    for i in range(k):
        H[0,i]=entropy[data.y[:,0]==i].mean()
    a=(torch.max(H, dim=1)[0]/torch.min(H, dim=1)[0])
    return print(H,a)


def plot1(y_true, y_pred, y_pro, entropy, args):
    """
    Generate scatter plot showcasing the classification results.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - y_pro: Probabilities of being predicted as the majority class.
    - entropy: Geometric entropy values.
    - args: Object containing dataset, method, and mixup attributes.

    Returns:
    None. Saves the plot to a file.
    """
    
    # Find TP, TN, FP, FN based on y_true and y_pred
    TP=[i for i, (a, p) in enumerate(zip(y_true, y_pred)) if a == 0 and p == 0]
    TN=[i for i, (a, p) in enumerate(zip(y_true, y_pred)) if a == 1 and p == 1]
    FP=[i for i, (a, p) in enumerate(zip(y_true, y_pred)) if a == 1 and p == 0]
    FN=[i for i, (a, p) in enumerate(zip(y_true, y_pred)) if a == 0 and p == 1]
    entropy=np.array(entropy)
    y_pro=np.array(y_pro)
    # print(y_pro)

    # p=np.nonzero(y_true==1)[0].shape[0]
    # n=np.nonzero(y_true==0)[0].shape[0]
    
    fig, ax = plt.subplots()
    plt.rcParams["font.family"] = "Times New Roman"
    
    # ax.scatter(entropy, y_true, color='red', marker='o', s=10, label='Training Data', alpha=0.1)
    ax.scatter(entropy[TP], y_pro[TP], color='deepskyblue', marker='o', s=10, label='TP', alpha=0.5)
    ax.scatter(entropy[TN], y_pro[TN], color='lime', marker='o', s=10, label='TN', alpha=0.5)
    ax.scatter(entropy[FP], y_pro[FP], color='darkblue', marker='*', s=80, label='FP', alpha=0.5)
    ax.scatter(entropy[FN], y_pro[FN], color='darkgreen', marker='*', s=80, label='FN', alpha=0.5)
    
    ax.set_ylabel('Probability of Predicted as Majority Class', fontname='Times New Roman', fontsize=15)
    ax.set_xlabel('Geometric Entropy', fontname='Times New Roman', fontsize=15)
    ax.legend(loc='lower left')
    
    plt.savefig('./image/' + args.dataset + '_' + args.method + '_' + str(args.mixup) + '.png')
    plt.close(fig)
    
    accuracy = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, average = 'weighted')
    f1_macro = f1_score(y_true, y_pred, average = 'macro')
    
    print(f'Accuracy: {accuracy:.4f}, Balanced Accuracy: {bacc:.4f}, Weighted F1: {w_f1:.4f}, Macro F1: {f1_macro:.4f}')
    
def plot_results(y_true, y_pred, entropy, args):

    # Convert tensors to numpy arrays for plotting.
    entropy = np.array(entropy)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    log_entropy = entropy

    # Define 5 evenly spaced intervals based on the range of log entropy values
    n = 5
    min_entropy, max_entropy = log_entropy.min(), log_entropy.max()
    range_expansion = (max_entropy - min_entropy) * 0.01
    interval_boundaries = np.linspace(min_entropy - range_expansion, max_entropy + range_expansion, n + 1)
    intervals = [(interval_boundaries[i], interval_boundaries[i + 1]) for i in range(n)]

    labels = [f"{i[0]:.2f}-{i[1]:.2f}" for i in intervals]

    total_counts_per_interval = []
    total_accuracies_per_interval = []
    # Calculate accuracy for each interval
    for start, end in intervals:
        mask = (log_entropy >= start) & (log_entropy < end)
        total_count = mask.sum()
        total_counts_per_interval.append(total_count)

        correct_preds = y_true[mask] == y_pred[mask]
        correct_preds_mean = np.around(correct_preds.mean(), decimals=3) if total_count > 0 else 0
        total_accuracies_per_interval.append(correct_preds_mean)

    # Bar plot for total counts and accuracies
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Total Counts Plot
    ax1.bar(range(n), total_counts_per_interval, color='blue')
    ax1.set_xlabel('Degree', fontname='Times New Roman', fontsize=15)
    # ax1.set_xlabel('Local Homophily Ratio', fontname='Times New Roman', fontsize=15)
    ax1.set_ylabel('Total Counts', fontname='Times New Roman', fontsize=15)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(labels)

    # Total Accuracies Plot
    ax2.bar(range(n), total_accuracies_per_interval, color='blue')
    ax2.set_xlabel('Degree', fontname='Times New Roman', fontsize=15)
    # ax2.set_xlabel('Local Homophily Ratio', fontname='Times New Roman', fontsize=15)
    ax2.set_ylabel('Total Accuracy', fontname='Times New Roman', fontsize=15)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(labels)

    plt.tight_layout()
    plt.savefig('./image/degree/combined_degree_20.png')
    plt.close(fig)

def calculate_accuracy(true, pred):
    return np.sum(true == pred) / len(true) if len(true) > 0 else 0

def calculate_std(true, pred):
    """Calculate the standard deviation of the accuracies."""
    return np.std(true == pred) if len(true) > 0 else 0

def calculate_precision(true, pred):
    tp = np.sum((pred == 1) & (true == pred))
    fp = np.sum((pred == 1) & (true != pred))
    return tp / (tp + fp) if tp + fp > 0 else 0

def calculate_recall(true, pred):
    tp = np.sum((pred == 1) & (true == pred))
    fn = np.sum((pred == 0) & (true != pred))
    return tp / (tp + fn) if tp + fn > 0 else 0

def calculate_f1_score(true, pred):
    precision = calculate_precision(true, pred)
    recall = calculate_recall(true, pred)
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

def calculate_accuracy(true, pred):
    return np.sum(true == pred) / len(true) if len(true) > 0 else 0

def plot_edge(y_true, y_pred, entropy, label, entro):

    # Convert tensors to numpy arrays for plotting.
    # value = np.array(entropy)
    value = np.array(entro)
    label = np.array(label)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Assuming label is a numpy array or a list
    # num_label_1 = np.sum(label == 1)
    # print(f"Number of instances where label == 1: {num_label_1}")

    num_bars = 5
    accuracy_data = {0: {'distrust': [], 'trust': []}, 1: {'distrust': [], 'trust': []}, 2: {'distrust': [], 'trust': []}, 
                    3: {'distrust': [], 'trust': []}, 4: {'distrust': [], 'trust': []}, 5: {'distrust': [], 'trust': []}}
    std_dev_data = {0: {'distrust': [], 'trust': []}, 1: {'distrust': [], 'trust': []}, 2: {'distrust': [], 'trust': []}, 
                    3: {'distrust': [], 'trust': []}, 4: {'distrust': [], 'trust': []}, 5: {'distrust': [], 'trust': []}}
    value_ranges = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    count_data = {0: {'distrust': [], 'trust': []}, 1: {'distrust': [], 'trust': []}, 2: {'distrust': [], 'trust': []}, 
                    3: {'distrust': [], 'trust': []}, 4: {'distrust': [], 'trust': []}, 5: {'distrust': [], 'trust': []}}
    
    avg_acc_trust = []
    avg_acc_distrust = []

    for label_value in [0, 1, 2, 3, 4, 5]:
        mask = label == label_value
        label_values = value[mask]
        true_labels = y_true[mask]
        predicted_labels = y_pred[mask]
        
        mask_distrust = true_labels == 1
        mask_trust = true_labels == 0

        # Calculate accuracy for 'distrust' and 'trust'
        # if mask_distrust.any():  # Check if 'distrust' cases exist
        #     accuracy_distrust = accuracy_score(true_labels[mask_distrust], predicted_labels[mask_distrust])
        # else:
        #     accuracy_distrust = 0
            
        # if mask_trust.any():  # Check if 'trust' cases exist
        #     accuracy_trust = accuracy_score(true_labels[mask_trust], predicted_labels[mask_trust])
        # else:
        #     accuracy_trust = 0 # No 'trust' cases to calculate accuracy

        # # Calculate F1 score for 'trust' and 'distrust' within the current label
        if len(np.unique(true_labels)) > 1:  # Check if both classes are present
            f1_distrust = f1_score(true_labels[mask_distrust], predicted_labels[mask_distrust], pos_label=1)
            f1_trust = f1_score(true_labels[mask_trust], predicted_labels[mask_trust], pos_label=0)
            f1_marco = f1_score(true_labels, predicted_labels, average = 'macro')
        else:
            f1_distrust = 0 if 1 not in true_labels else f1_score(true_labels[mask_distrust], predicted_labels[mask_distrust], pos_label=1)
            f1_trust = 0 if 0 not in true_labels else f1_score(true_labels[mask_trust], predicted_labels[mask_trust], pos_label=0)
            f1_marco = f1_score(true_labels, predicted_labels, average = 'macro')

        if len(label_values) > 0:
            value_min, value_max = label_values.min(), label_values.max()
            value_min, value_max = value_min - 0.01 * (value_max - value_min), value_max + 0.01 * (value_max - value_min)
            value_intervals = np.linspace(value_min, value_max, num_bars + 1)

            for i in range(num_bars):
                value_mask = (value >= value_intervals[i]) & (value <= value_intervals[i+1]) if i == num_bars - 1 else (value >= value_intervals[i]) & (value < value_intervals[i+1])
                combined_mask = mask & value_mask
                
                distrust_mask = combined_mask & (y_true == 1)
                trust_mask = combined_mask & (y_true == 0)

                # Count quantity
                count_distrust = np.sum(distrust_mask)
                count_trust = np.sum(trust_mask)

                count_data[label_value]['distrust'].append(count_distrust)
                count_data[label_value]['trust'].append(count_trust)

                # Calculate accuracy for distrust and trust
                # acc_distrust = calculate_accuracy(y_true[distrust_mask], y_pred[distrust_mask])
                # acc_trust = calculate_accuracy(y_true[trust_mask], y_pred[trust_mask])

                acc_distrust = calculate_f1_score(y_true[distrust_mask], y_pred[distrust_mask])
                acc_trust = calculate_f1_score(y_true[trust_mask], y_pred[trust_mask])
                # std_dev_distrust = calculate_std(y_true[combined_mask], y_pred[combined_mask] == 1)
                # std_dev_trust = calculate_std(y_true[combined_mask], y_pred[combined_mask] == 0)
                # std_dev_data[label_value]['distrust'].append(std_dev_distrust)
                # std_dev_data[label_value]['trust'].append(std_dev_trust)
                
                accuracy_data[label_value]['distrust'].append(acc_distrust)
                accuracy_data[label_value]['trust'].append(acc_trust)
                value_ranges[label_value].append(f'{value_intervals[i]:.2f}-{value_intervals[i+1]:.2f}')
        
        # avg_acc_distrust.append(accuracy_distrust)
        # avg_acc_trust.append(accuracy_trust)
        avg_acc_distrust.append(f1_distrust)
        avg_acc_trust.append(f1_trust)
        # avg_acc_distrust.append(f1_marco)
        # avg_acc_trust.append(f1_marco)

    # Plotting
    for i in range(6):
        print(f'Plot {i+1} - Average Accuracy distrust: {avg_acc_distrust[i]:.2f}, Average Accuracy trust: {avg_acc_trust[i]:.2f}')
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.rcParams["font.family"] = "Times New Roman"
    axes = axes.flatten()
    bar_width = 0.35  # Width of the bars

    for label_value, ax in zip([0, 1, 2, 3, 4, 5], axes):
        distrust_accuracies = accuracy_data[label_value]['distrust']
        trust_accuracies = accuracy_data[label_value]['trust']
        # for i, (acc_distrust, acc_trust) in enumerate(zip(distrust_accuracies, trust_accuracies)):
        #     print(f"Plot {label_value+1}, Bar {i+1} - Distrust Accuracy: {acc_distrust:.2f}, Trust Accuracy: {acc_trust:.2f}")
            
        distrust_counts = np.array(count_data[label_value]['distrust']).sum()
        trust_counts = np.array(count_data[label_value]['trust']).sum()
        print(f"Plot {label_value+1} - Distrust Counts: {distrust_counts}, Trust Counts: {trust_counts}")
        # print(value_ranges[label_value])
        indices = np.arange(len(accuracy_data[label_value]['trust']))
        # ax.bar(indices - bar_width/2, accuracy_data[label_value]['distrust'], bar_width, yerr=std_dev_data[label_value]['distrust'], label='Distrust', alpha=0.7, capsize=5)
        # ax.bar(indices + bar_width/2, accuracy_data[label_value]['trust'], bar_width, yerr=std_dev_data[label_value]['trust'], label='Trust', alpha=0.7, capsize=5)
        ax.bar(indices - bar_width/2, accuracy_data[label_value]['distrust'], bar_width, label='Minority', alpha=0.7, capsize=5)
        ax.bar(indices + bar_width/2, accuracy_data[label_value]['trust'], bar_width, label='Majority', alpha=0.7, capsize=5)
        
        max_quantity = max(max(count_data[label_value]['distrust']), max(count_data[label_value]['trust']))
        scaled_distrust = [-x / max_quantity for x in count_data[label_value]['distrust']]
        scaled_trust = [-x / max_quantity for x in count_data[label_value]['trust']]
        ax.bar(indices - bar_width/2, scaled_distrust, bar_width, color ='black', alpha=0.7)
        ax.bar(indices + bar_width/2, scaled_trust, bar_width, color ='black', alpha=0.7)
        
        ax.set_ylim(-1.1, 1.1)
        ax.set_xticks(indices)
        ax.set_xticklabels(value_ranges[label_value], rotation=45)
        ax.set_xlabel('GE Value')
        ax.set_ylabel('Training Accuracy / Scaled Quantity (×{})'.format(max_quantity))
        ax.legend()

    plt.tight_layout()
    plt.savefig('./image/quan_cate/epinions_0.6_test.png')
    plt.close(fig)

def plot_classification_results(y_true, y_pred, entropy, args):
    """
    Generate bar plot showcasing the classification results per entropy interval.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - entropy: Geometric entropy values.
    - args: Object containing dataset, method, and mixup attributes.

    Returns:
    None. Saves the plot to a file.
    """

    # Convert tensors to numpy arrays for plotting.
    entropy = np.array(entropy)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Define 5 evenly spaced intervals based on the range of entropy values
    # n = 5
    # min_entropy, max_entropy = entropy.min(), entropy.max()
    # # Expanding the range slightly
    # range_expansion = (max_entropy - min_entropy) * 0.01
    # interval_boundaries = np.linspace(min_entropy - range_expansion, max_entropy + range_expansion, n + 1)
    # intervals = [(interval_boundaries[i], interval_boundaries[i + 1]) for i in range(n)]
    
    # entropy += np.abs(entropy.min()) + 1e-8  # Adding a small constant to avoid log(0)
    # log_entropy = np.log(entropy)  # Applying logarithm


    # Define 5 evenly spaced intervals based on the range of log entropy values
    n = 5
    min_entropy, max_entropy = entropy.min(), entropy.max()
    log_entropy = (entropy - min_entropy) / (max_entropy - min_entropy)
    min_entropy, max_entropy = log_entropy.min(), log_entropy.max()
    range_expansion = (max_entropy - min_entropy) * 0.01
    interval_boundaries = np.linspace(min_entropy - range_expansion, max_entropy + range_expansion, n + 1)
    intervals = [(interval_boundaries[i], interval_boundaries[i + 1]) for i in range(n)]

    labels = [f"{i[0]:.2f}-{i[1]:.2f}" for i in intervals]
    unique_classes = np.unique(y_true)
    
    counts_per_interval = []
    for start, end in intervals:
        # mask = (entropy >= start) & (entropy < end)
        mask = (log_entropy >= start) & (log_entropy < end)
        counts = [np.sum(y_true[mask] == cls) for cls in unique_classes]
        counts_per_interval.append(counts)

    print(counts_per_interval)  # Print results in desired format
    
    class_accuracies = {cls: [] for cls in unique_classes}

    # Calculate accuracy for each interval
    for start, end in intervals:
        # mask = (entropy >= start) & (entropy < end)
        mask = (log_entropy >= start) & (log_entropy < end)

        if mask.sum() == 0:  # if no data in this interval
            for cls in unique_classes:
                class_accuracies[cls].append(0)
            continue

        for cls in unique_classes:
            correct_preds = np.logical_and(y_true[mask] == cls, y_pred[mask] == cls)
            correct_preds_mean = np.around(correct_preds.mean(), decimals=3)
            class_accuracies[cls].append(correct_preds_mean)
            
    # print(class_accuracies)

    fig, ax = plt.subplots()
    plt.rcParams["font.family"] = "Times New Roman"

    # Bar plot
    barWidth = 1 / (len(unique_classes) + 1)  # calculate dynamic width based on number of classes
    r_values = [np.arange(len(labels)) + barWidth * i for i in range(len(unique_classes))]

    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']  # more colors can be added if necessary
    for i, cls in enumerate(unique_classes):
        ax.bar(r_values[i], class_accuracies[cls], width=barWidth, color=colors[i], label=f'Class {cls}')

    # Labels and legends.
    # ax.set_xlabel('Deviation Weight', fontname='Times New Roman', fontsize=15)
    ax.set_xlabel('Weighted Local Homophily Ratio', fontname='Times New Roman', fontsize=15)
    ax.set_ylabel('Training Accuracy', fontname='Times New Roman', fontsize=15)
    ax.set_xticks([r + barWidth * (len(unique_classes) / 2) for r in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')

    # Save figure.
    if args.task == 0:
        plt.tight_layout()
        plt.savefig('./image/weighted_homo/' + args.dataset + '_' + args.method + '_' + str(args.n) + '_' + str(args.w) + '_bar_class.png')
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.savefig('./image/weighted_homo/' + args.dataset + '_' + args.method + '_' + str(args.n) + '_' + str(args.w) + '_multi_bar.png')
        plt.close(fig)


def edge_emb(x, edge_index, edge_idx):
    head, tail = edge_index[:, edge_idx][0], edge_index[:, edge_idx][1]
    head_emb = x[head]
    tail_emb = x[tail]
    
    edge_emb = torch.cat([head_emb, tail_emb], dim=1)
    return edge_emb

def concatedge(a, b):
    # Identify common values
    common_values = torch.where(a[0] == b[0], a[0], 
                    torch.where(a[0] == b[1], a[0], 
                    torch.where(a[1] == b[0], a[1], 
                    torch.where(a[1] == b[1], a[1], 0))))
    # Identify values in a that are not common
    a_non_match = torch.where(common_values != a[0], a[0], a[1])
    # Identify values in b that are not common
    b_non_match = torch.where(common_values != b[0], b[0], b[1])
    result = torch.stack([common_values, a_non_match, b_non_match])
    return result

def compute_mixup(x, edges_to_match, found_edges, lam):
    concat_edge = concatedge(edges_to_match, found_edges)

    head = concat_edge[0]
    tail1 = concat_edge[1]
    tail2 = concat_edge[2]
    
    head_emb = x[head]
    tail1_emb = x[tail1]
    tail2_emb = x[tail2]
    
    edge_emb2 = torch.cat([head_emb, lam*tail1_emb+(1-lam)*tail2_emb], dim=1)
    return edge_emb2

def compute_mixup2(x, edges_to_match, found_edges, lam):
    concat_edge = concatedge(edges_to_match, found_edges)

    head = concat_edge[0]
    tail1 = concat_edge[1]
    tail2 = concat_edge[2]
    
    head_emb = x[head]
    tail1_emb = x[tail1]
    tail2_emb = x[tail2]
    
    edge_emb1 = torch.cat([head_emb, tail1_emb], dim=1)
    edge_emb2 = torch.cat([head_emb, tail2_emb], dim=1)
    return edge_emb1, edge_emb2

def get_unique_counts(tensor):
    tensor = tensor.view(-1)  # Flatten tensor to 1D
    min_val = tensor.min()
    
    if min_val < 0:
        tensor -= min_val  # Offset tensor values to make them non-negative
    
    unique_values = torch.unique(tensor)
    counts = torch.bincount(tensor)
    return dict(zip((unique_values + min_val).tolist(), counts[unique_values].tolist()))
    
def mixup(x, data, idx1, idx2, train_edge0, args):
    
    if args.mixup == 1:
        lam = np.random.beta(args.alpha, args.alpha)
        K=int(len(idx2)*args.k)
        perm = torch.randperm(len(idx2))
        edge_list = perm[:K]
        selected_edges = idx1[edge_list]
        
        # edge_list = torch.topk(entropy, K).indices

        # # Select edges from edge_list
        # selected_edges = idx1[edge_list]
        edges_to_match = data.edge_index[:, selected_edges]
        prop_head = train_edge0[0].unsqueeze(1)
        prop_tail = train_edge0[1].unsqueeze(1)
                
        connected_mask_head = (prop_head == edges_to_match[0]) | (prop_head == edges_to_match[1])
        connected_mask_tail = (prop_tail == edges_to_match[0]) | (prop_tail == edges_to_match[1])
        
        connected_mask = connected_mask_head | connected_mask_tail
        exclude_mask = (prop_head == edges_to_match[0]) & (prop_tail == edges_to_match[1])
        connected_mask = connected_mask & ~exclude_mask

        connected_indices = connected_mask.long().argmax(dim=0)
        found_edges = train_edge0[:, connected_indices]

        matching = (data.edge_index.unsqueeze(2) == found_edges.unsqueeze(1)).all(dim=0)
        edge_list2 = matching.long().argmax(dim=0)
        
        # edge_emb2 = compute_mixup(x, edges_to_match, found_edges, lam)
        edge_emb1, edge_emb2 = compute_mixup2(x, edges_to_match, found_edges, lam)
    
    if args.mixup == 2:
        lam = np.random.beta(args.alpha, args.alpha)
        entropy = args.topo_reweight[idx2]
        K=int(len(idx2)*args.k)
        probabilities = F.softmax(entropy, dim=0)
        edge_list = torch.multinomial(probabilities, K, replacement=False)

        # Select edges from edge_list
        selected_edges = idx1[edge_list]
        edges_to_match = data.edge_index[:, selected_edges]
        prop_head = train_edge0[0].unsqueeze(1)
        prop_tail = train_edge0[1].unsqueeze(1)
                
        connected_mask_head = (prop_head == edges_to_match[0]) | (prop_head == edges_to_match[1])
        connected_mask_tail = (prop_tail == edges_to_match[0]) | (prop_tail == edges_to_match[1])
        
        connected_mask = connected_mask_head | connected_mask_tail
        exclude_mask = (prop_head == edges_to_match[0]) & (prop_tail == edges_to_match[1])
        connected_mask = connected_mask & ~exclude_mask

        # connected_indices = connected_mask.long().argmax(dim=0)
        connected_indices = (args.topo_reweight[:, None] * connected_mask.long()).argmax(dim=0)
        found_edges = train_edge0[:, connected_indices]

        matching = (data.edge_index.unsqueeze(2) == found_edges.unsqueeze(1)).all(dim=0)
        edge_list2 = matching.long().argmax(dim=0)
        
        edge_emb2 = compute_mixup(x, edges_to_match, found_edges, lam)
    return edge_emb2, edge_list, edge_list2, connected_indices, lam
    #     edge_emb1, edge_emb2 = compute_mixup2(x, edges_to_match, found_edges, lam)
    # return edge_emb1, edge_emb2, edge_list, edge_list2, connected_indices, lam


def edge_adj(edge_index, num_edges):
    edge_node_matrix = torch.zeros((num_edges, edge_index.max() + 1), dtype=torch.bool)
    edge_node_matrix.scatter_(1, edge_index.t(), 1)
    edge_node_matrix_float = edge_node_matrix.float()
    edge_adj_matrix = torch.matmul(edge_node_matrix_float, edge_node_matrix_float.t())
    edge_adj_matrix = edge_adj_matrix.bool()
    edge_adj_matrix.fill_diagonal_(0)

    return edge_adj_matrix


def local_homo_ratio(edge_index, edge_labels, num_hops, w):
    num_edges = edge_index.size(1)
    edge_adj_matrix = edge_adj(edge_index, num_edges)
    ratios = torch.zeros(num_edges)
    norm_factor = sum(w**k for k in range(1, num_hops+1))

    # Initial adjacency matrix is the 1-hop neighborhood
    accumulated_adj = edge_adj_matrix.clone()
    
    for n in range(1, num_hops + 1):
        # Apply decay factor to the adjacency matrix
        decay_factor = (w ** n)/norm_factor
        for i in range(num_edges):
            edge_label = edge_labels[i]
            n_hop_neighbors = accumulated_adj[i]
            n_hop_neighbors[i] = False  # Exclude the original edge

            # Count same-label edges
            same_label_count = edge_labels[n_hop_neighbors][edge_labels[n_hop_neighbors] == edge_label].size(0)
            total_count = n_hop_neighbors.sum().item()

            # Compute weighted ratio
            if total_count > 0:
                ratio = decay_factor * (same_label_count / total_count)
                ratios[i] += ratio
        
        # Update the adjacency matrix if not the last hop
        if n < num_hops:
            accumulated_adj = torch.matmul(accumulated_adj.float(), edge_adj_matrix.float()).bool()

    return ratios/num_hops

def densityplt(result):
    # Determine the number of classes and feature dimensions
    num_classes = len(result[0])
    feature_dim = num_classes  # Assuming feature_dim is same as num_classes

    fig, axes = plt.subplots(num_classes, feature_dim, figsize=(6*num_classes, 6*num_classes))

    for cls in range(num_classes):
        for dim in range(feature_dim):
            ax = axes[cls, dim]
            values = torch.cat([simulation[cls][:,dim] for simulation in result]).numpy()
            # values = [simulation[cls][dim] for simulation in result]
            # # Ensure each value is at least 1D tensor
            # values = [v.unsqueeze(0) if v.dim() == 0 else v for v in values]
            # # Concatenate and convert to numpy
            # values = torch.cat(values).numpy()
            ax.hist(values, bins=40, range=(0, 1), color='teal')
            ax.set_title(f'Class {cls} - Dim {dim} - Mean: {values.mean():.4f}', fontsize = 20)
            ax.set_xlabel('Value', fontsize=20)  # Set X-axis label font size
            ax.set_ylabel('Frequency', fontsize=20) 
            ax.tick_params(axis='x', labelsize=20)  # Set x-axis tick label size for this particular subplot
            ax.tick_params(axis='y', labelsize=20)

    plt.tight_layout()
    plt.show()
    plt.savefig('./image/density/mc_binary.png', dpi = 200, transparent = True, bbox_inches = 'tight')
    
def mc(data, edge_idxs, train_y, num, args):
    result = []
    sum_pos_deg = 0
    sum_pos_deg2 = 0
    sum_neg_deg = 0
    sum_neg_deg2 = 0
    for i in range(num):
        perm = torch.randperm(train_y.size(0))
        train_y = train_y[perm]
        # args.entropy, pos_deg, pos_deg2, neg_deg, neg_deg2 = cal_topo_reweight(data.edge_index[:, edge_idxs['train']], train_y.view(-1), args)
        res = cal_topo_reweight(data.edge_index[:, edge_idxs['train']], train_y.view(-1), args)
        # result[i].append(args.entropy)
        result.append(res)

    #     sum_pos_deg += pos_deg
    #     sum_pos_deg2 += pos_deg2
    #     sum_neg_deg += neg_deg
    #     sum_neg_deg2 += neg_deg2
    # sum_first_tensor = torch.zeros_like(result[0][0][0])
    # sum_second_tensor = torch.zeros_like(result[0][0][1])
    # for key in result:
    #     sum_first_tensor += result[key][0][0]
    #     sum_second_tensor += result[key][0][1]
    # mean_first_tensor = sum_first_tensor / len(result)
    # mean_second_tensor = sum_second_tensor / len(result)
    
    # mean_pos_deg = sum_pos_deg / num
    # mean_pos_deg2 = sum_pos_deg2 / num
    # mean_neg_deg = sum_neg_deg / num
    # mean_neg_deg2 = sum_neg_deg2 / num
    
    # print(mean_pos_deg, mean_pos_deg2, mean_neg_deg, mean_neg_deg2)
    # print("num",num)
    # print("Mean of the Positive Encoding:", mean_first_tensor)
    # print("Mean of the Negative Encoding:", mean_second_tensor)
    # print("==========")
    # return result, mean_first_tensor, mean_second_tensor
    return result

def ge_reweight(edge_index, edge_attr, args):

    edge_index2, edge_attr2 = process_edge_attr(edge_index, edge_attr)
    num_classes = torch.unique(edge_attr2).shape[0]
    max_edge_attr = edge_attr2.max().item()
    edge_one_hot = torch.zeros(edge_attr2.shape[0], max(max_edge_attr + 1, num_classes))
    # edge_one_hot = torch.zeros(edge_attr2.shape[0], torch.unique(edge_attr2).shape[0])
    edge_one_hot[torch.arange(edge_attr2.shape[0]), edge_attr2] = 1
    node_one_hot = scatter_add(edge_one_hot, edge_index2[0], dim = 0)/degree(edge_index2[0]).view(-1, 1)
    # edge_index2, _ = add_self_loops(edge_index2, num_nodes = edge_index2.max().item() + 1)
    node_deg = degree(edge_index2[0])
    deg_inv_sqrt = 1/torch.sqrt(node_deg)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    # edge_weight = deg_inv_sqrt[edge_index2[0]] * deg_inv_sqrt[edge_index2[1]]
    edge_weight = deg_inv_sqrt[edge_index2[0]] * deg_inv_sqrt[edge_index2[0]]
    adj_t = torch.sparse_coo_tensor(edge_index2, edge_weight, torch.Size([node_deg.shape[0], node_deg.shape[0]]))
    
    # Method 1: A = A^n
    adj_power = adj_t
    for _ in range(1, args.n):
        adj_power = torch.sparse.mm(adj_power, adj_t)
    diff_node_one_hot = torch.sparse.mm(adj_power, node_one_hot)
    
    # # Method 2: A = 1/k*(A+A^2+...+A^K)
    # adj_power = adj_t
    # adj_sum = adj_t.clone()
    # for _ in range(1, args.n):
    #     adj_power = torch.sparse.mm(adj_power, adj_t)
    #     adj_sum += adj_power
    # adj_avg = adj_sum / args.n
    # diff_node_one_hot = torch.sparse.mm(adj_avg, node_one_hot)
    
    # # Method 2: A = betaA+(betaA)^2+...+(betaA)^K
    # beta = args.beta
    # adj_power = beta * adj_t  # Scale the adjacency matrix by beta
    # adj_sum = adj_power.clone()  # Initialize adj_sum with beta * adj_t
    # for _ in range(1, args.n):
    #     adj_power = torch.sparse.mm(adj_power, adj_t) * beta  # Multiply by beta at each power
    #     adj_sum += adj_power
    # diff_node_one_hot = torch.sparse.mm(adj_sum, node_one_hot)
    
    diff_edge_one_hot = (diff_node_one_hot[edge_index[0]] + diff_node_one_hot[edge_index[1]])/2
    # edge_attr = edge_attr.reshape(-1,1)
    # diff_edge_one_hot = torch.where(edge_attr == 0, diff_edge_one_hot - first, diff_edge_one_hot - second)
    
    dist = Categorical(probs = diff_edge_one_hot).entropy()
    # weight=dist
    # dist = 1/torch.norm(diff_edge_one_hot, p=2, dim=1)
    weight = t_sof(dist, args)
    # weight = torch.norm(diff_edge_one_hot, p=2, dim=1)
    # weight = torch.log(torch.norm(diff_edge_one_hot, p=2, dim=1) + 1e-8)

    return weight

def t_sof(y, args):
    T=args.T
    scaled_y = y / T
    p = torch.exp(scaled_y) 
    # p = F.softmax(scaled_y, dim=-1)
    # p = torch.sigmod(scaled_y)
    return p

def calculate_edge_ratios(edge_index, edge_labels):

    # Initialize counters for each node
    pos_adj_count = torch.zeros(edge_index.max() + 1)
    neg_adj_count = torch.zeros_like(pos_adj_count)

    # Count adjacent positive and negative edges for each node
    for src, dest in edge_index.t():
        label = edge_labels[src]
        pos_adj_count[src] += (label == 0)
        neg_adj_count[src] += (label == 1)

    # Initialize sum counters for the ratios
    sum_ratios = torch.zeros(4)

    for src, dest in edge_index.t():
        label = edge_labels[src]
        total_edges_src = pos_adj_count[src] + neg_adj_count[src]
        total_edges_dest = pos_adj_count[dest] + neg_adj_count[dest]

        if label == 0:  # Positive edge
            sum_ratios[0] += pos_adj_count[src] / total_edges_src if total_edges_src > 0 else 0
            sum_ratios[1] += pos_adj_count[dest] / total_edges_dest if total_edges_dest > 0 else 0
        else:  # Negative edge
            sum_ratios[2] += pos_adj_count[src] / total_edges_src if total_edges_src > 0 else 0
            sum_ratios[3] += pos_adj_count[dest] / total_edges_dest if total_edges_dest > 0 else 0

    # Calculate the averages
    num_pos_edges = (edge_labels == 0).sum().item()
    num_neg_edges = (edge_labels == 1).sum().item()
    ratios = sum_ratios / torch.tensor([num_pos_edges, num_pos_edges, num_neg_edges, num_neg_edges])

    return ratios[0].item(), ratios[1].item(), ratios[2].item(), ratios[3].item()

    
# =====ECN=====
def jaccard_coefficient(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def compute_similarity(train_edge_index, train_y, num_classes, num_nodes, k):
    # Initialize neighbor sets for each class and each node
    I = [[set() for _ in range(num_classes)] for _ in range(num_nodes)]

    # Populate the sets considering edges as undirected
    for i, j, label in zip(train_edge_index[0], train_edge_index[1], train_y):
        label = label.item()
        I[i][label].add(j)
        I[j][label].add(i)

    # Compute Jaccard coefficients and overall similarity
    top_k_similar = torch.zeros((num_nodes, k), dtype=torch.int64)
    for node in range(num_nodes):
        jaccard_scores = torch.zeros(num_nodes)
        for other_node in range(num_nodes):
            if node != other_node:
                for c in range(num_classes):
                    jaccard_scores[other_node] += jaccard_coefficient(I[node][c], I[other_node][c])

        # Find top-k similar nodes
        _, top_k_nodes = torch.topk(jaccard_scores, k)
        top_k_similar[node] = top_k_nodes

    return top_k_similar


def find_similar_nodes_for_test_edges(train_data, test_data, k, mu):
    # Compute top-k similar vertices using the training data
    top_k_similar = compute_similarity(train_data, k, mu)

    # For each edge in the test data, find the most similar nodes
    similar_nodes_for_test_edges = []
    for edge in test_data.edge_index.t():
        u, v = edge.tolist()
        similar_nodes_u = top_k_similar[u]
        similar_nodes_v = top_k_similar[v]
        similar_nodes_for_test_edges.append((similar_nodes_u, similar_nodes_v))

    return similar_nodes_for_test_edges

def edge_classification_and_metrics(train_edge_index, train_y, test_edge_index, test_y, top_k_similar, k, num_classes, num_nodes):

    # Initialize neighbor sets for each class and each node
    I = [[set() for _ in range(num_classes)] for _ in range(num_nodes)]  # num_nodes must be known

    # Populate the sets based on edge labels in the training data
    for i, j, label in zip(train_edge_index[0], train_edge_index[1], train_y):
        I[i][label.item()].add(j)
        I[j][label.item()].add(i)

    edge_predictions = []
    for edge in test_edge_index.t():
        u, v = edge.tolist()

        # Determine the most similar vertices for u and v in the test graph
        S_u = top_k_similar[u][:k]
        S_v = top_k_similar[v][:k]

        # Find the dominant class label among the similar vertices
        label_counts = Counter()
        for c in range(num_classes):
            J = jaccard_coefficient(I[u][c], I[v][c])
            label_counts[c] = J

        # Predict the class with the highest Jaccard coefficient
        predicted_class = label_counts.most_common(1)[0][0]
        edge_predictions.append(predicted_class)

    # Convert predictions to a tensor
    predicted_labels = torch.tensor(edge_predictions)

    # Compute metrics
    balanced_accuracy = balanced_accuracy_score(test_y.numpy(), predicted_labels.numpy())
    macro_f1 = f1_score(test_y.numpy(), predicted_labels.numpy(), average='macro')

    return balanced_accuracy, macro_f1

def class_wise_accuracy(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    num_classes = torch.max(y_true).item() + 1
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    for i in range(num_classes):
        class_mask = (y_true == i)
        class_correct[i] = torch.sum((y_pred[class_mask] == y_true[class_mask]))
        class_total[i] = torch.sum(class_mask)

    class_accuracies = class_correct / class_total
    return class_accuracies

# def class_wise_f1(y_true, y_pred):
#     y_true = torch.tensor(y_true)
#     y_pred = torch.tensor(y_pred)
#     num_classes = torch.max(y_true).item() + 1
#     class_f1 = torch.zeros(num_classes)

#     for i in range(num_classes):
#         class_mask = (y_true == i)
#         class_f1[i] = f1_score(y_true[class_mask], y_pred[class_mask])
        
#     return class_f1

def class_f1(cm):
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    
    return f1_scores
        
    

