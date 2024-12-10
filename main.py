from param_parser import parse_args
from utils import *
import os
import torch
from dataset import load_dataset, split_edge, EdgeDataset
from torch_geometric.loader import DataLoader
from model import  *
import torch.optim as optim
import torch.nn as nn
from torch_geometric.nn import Node2Vec
from learn import *
import math
from sentence_transformers import SentenceTransformer
import time
import numpy as np
from sklearn.manifold import TSNE
import warnings

# To ignore all user warnings
warnings.filterwarnings("ignore", category=UserWarning)

def run(data, loaders, model, classifier, optimizer, loss_fn, train_edge0, train_edge, args):
    best_val_f1_macro = -math.inf
    best_epoch = 0
    patience = 20  # Number of epochs to wait before early stopping if no improvement
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train(data, model, classifier, loaders['train'], optimizer, loss_fn, train_edge0, train_edge, args)

        if epoch % 5 == 0:
            val_acc, bacc, f1, f1_macro, f1_micro, _ , _, _= eval(data, model, classifier, loaders['val'], train_edge, args)

            if f1_macro > best_val_f1_macro:
                # print('Epoch: {:03d}, Val Accuracy: {:.4f}, Balanced Accuracy: {:.4f}, F1: {:.4f}, F1 Macro: {:.4f}, F1 Micro: {:.4f}'\
                #     .format(epoch, val_acc, bacc, f1, f1_macro, f1_micro))
                best_val_f1_macro = f1_macro
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), f'./model/{args.dataset}/best_model.pt')
                torch.save(classifier.state_dict(), f'./model/{args.dataset}/best_classifier.pt')
                
            else:
                patience_counter += 1
                if patience_counter == patience:
                    break
    
    model.load_state_dict(torch.load(f'./model/{args.dataset}/best_model.pt'))
    classifier.load_state_dict(torch.load(f'./model/{args.dataset}/best_classifier.pt'))
    
    test_acc, bacc, f1, f1_macro, f1_micro, cm, y_true, y_pred = eval(data, model, classifier, loaders['test'], train_edge, args) #train_val_edge
    return test_acc, bacc, f1, f1_macro, f1_micro


if __name__ == '__main__':
    start = time.time()
    args = parse_args()
            
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()

    seed_everything(args.seed)
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device = args.device)

    data = load_dataset(args, encoder)
    edge_idxs = split_edge(data, args)

    res=[]

    train_edge0 = data.edge_index[:, edge_idxs['train']]
    
    train_val_edge = torch.cat([data.edge_index[:, edge_idxs['train']], data.edge_index[:, edge_idxs['val']]], dim = 1).to(args.device)
    train_edge = process_edge(train_edge0).to(args.device)
    train_edge, train_val_edge = train_edge.to(args.device), train_val_edge.to(args.device)
    train_edge0 = train_edge0.to(args.device)
    train_y = data.y[edge_idxs['train']]
    test_y = data.y[edge_idxs['test']]
    val_y = data.y[edge_idxs['val']]


    # # # topology reweight
    num_classes = int(torch.unique(data.y).shape[0])
    if args.method == 'tw' or args.mixup == 2:
        args.reweight = cal_reweight(train_y, num_classes).to(args.device)
        args.topo_reweight= ge_reweight(data.edge_index[:, edge_idxs['train']], train_y.view(-1), args).to(args.device)
    
    if args.method == 'qw':
        args.reweight = cal_reweight(train_y, num_classes)
        args.reweight = args.reweight.to(args.device)
        
    if args.method == 'tq':
        args.reweight = cal_reweight(train_y, num_classes)
        args.reweight = args.reweight.to(args.device)
        args.topo_reweight= ge_reweight(data.edge_index[:, edge_idxs['train']], train_y.view(-1), args).to(args.device) 
        args.w = (args.f * args.reweight[train_y] + (1 - args.f) * args.topo_reweight).to(args.device)
    
    split_data = {key: EdgeDataset(edge_idxs[key]) for key in edge_idxs}
    loaders = {}
    for key in split_data:
        if key == 'train':
            shuffle = True
        else:
            shuffle = False
        loaders[key] = DataLoader(split_data[key], batch_size = args.batch_size, \
                                shuffle = shuffle, collate_fn = split_data[key].collate_fn)
    
    for i in range(args.runs):
        data = data.to(args.device) 
        
        if args.model == 'GCN':
            model = GCNEncoder(data.num_nodes, args.n_embed, args.n_hidden).to(args.device)
        elif args.model == 'GAT':
            model = GATEncoder(data.num_nodes, args.n_embed, args.n_hidden).to(args.device)
        elif args.model == 'SAGE':
            model = SAGEEncoder(data.num_nodes, args.n_embed, args.n_hidden).to(args.device)
        elif args.model == 'Cheb':
            model = ChebEncoder(data.num_nodes, args.n_embed, args.n_hidden).to(args.device)
        else:
            raise NotImplementedError

        loss_fn = nn.CrossEntropyLoss(reduction = 'none')
        classifier = Classifier(args.n_hidden, data.edge_attr.shape[1], torch.unique(data.y).shape[0], args.dropout).to(args.device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=args.wd)
            
        test_acc, bacc, f1, f1_macro, f1_micro=run(data, loaders, model, classifier, optimizer, loss_fn, train_edge0, train_edge, args)
        
        result = {}   # Initialize an empty dictionary for this run's results
        result['test_acc'] = test_acc
        result['bacc'] = bacc
        result['f1'] = f1
        result['f1_macro'] = f1_macro
        result['f1_micro'] = f1_micro
        res.append(result)
        data = data.to('cpu')

    # metrics = ['test_acc', 'bacc', 'f1', 'f1_macro', 'f1_micro']
    metrics = [ 'bacc', 'f1_macro']
    means = {}
    stds = {}
    end = time.time()

    for metric in metrics:
        values = [r[metric] for r in res]
        means[metric] = np.mean(values)
        stds[metric] = np.std(values)
        
    metrics_output = []
    for metric in metrics:
        metrics_output.append("{}: {:.3f} $\\pm$ {:.3f}".format(metric, means[metric], stds[metric]))
        
    output_string = f"Total time: {end-start:.2f} seconds," + ', '.join(metrics_output)

    # Print the output to the console
    print(output_string)
