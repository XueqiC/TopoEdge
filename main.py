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
from learn import train, eval, evaltrain, evaltrain2
import math
from sentence_transformers import SentenceTransformer
import time
import wandb
import numpy as np
from sklearn.manifold import TSNE
import warnings

# To ignore all user warnings
warnings.filterwarnings("ignore", category=UserWarning)



def run(data, loaders, model, classifier, optimizer, loss_fn, train_edge0, train_edge, args):
    best_val_f1_macro = -math.inf
    best_epoch = 0
    
    for epoch in range(args.epochs):
        train(data, model, classifier, loaders['train'], optimizer, loss_fn, train_edge0, train_edge, args)

        if epoch % 5 == 0:
            val_acc, bacc, f1, f1_macro, f1_micro, _ , _, _= eval(data, model, classifier, loaders['val'], train_edge, args)

            if f1_macro > best_val_f1_macro:
                # print('Epoch: {:03d}, Val Accuracy: {:.4f}, Balanced Accuracy: {:.4f}, F1: {:.4f}, F1 Macro: {:.4f}, F1 Micro: {:.4f}'\
                #     .format(epoch, val_acc, bacc, f1, f1_macro, f1_micro))
                best_val_f1_macro = f1_macro
                best_epoch = epoch
                torch.save(model.state_dict(), f'./model/{args.dataset}/best_model.pt')
                torch.save(classifier.state_dict(), f'./model/{args.dataset}/best_classifier.pt')

    
    model.load_state_dict(torch.load(f'./model/{args.dataset}/best_model.pt'))
    classifier.load_state_dict(torch.load(f'./model/{args.dataset}/best_classifier.pt'))
    
    # y_true, y_predict, y_pro, entropy = evaltrain(data, model, classifier, loaders['train'], train_edge, args)
    # plot1(y_true, y_predict, y_pro, entropy, args)
    # plot_classification_results(y_true, y_predict, entropy, args)
    # plot_results(y_true, y_predict, entropy, args)
    
    # y_true, y_pred, y_pro, all, label, entro = evaltrain2(data, model, classifier, loaders['val'], train_edge, args)
    # plot_edge(y_true, y_pred, all, label, entro)
    
    test_acc, bacc, f1, f1_macro, f1_micro, cm, y_true, y_pred = eval(data, model, classifier, loaders['test'], train_edge, args) #train_val_edge
    # class_acc = class_wise_accuracy(y_true, y_pred)
    # print(class_acc[0], class_acc[1])
    # f1_class = class_f1(cm)
    # print(f1_class[0], f1_class[1], f1_class[0]-f1_class[1])
    
    # end = time.time()
    # print('Time: {:.4f},  Best epoch: {}, Test Accuracy: {:.4f}, Balanced Accuracy: {:.4f}, F1: {:.4f}, F1 Macro: {:.4f}, F1 Micro: {:.4f}'\
    #     .format(end-start, int(best_epoch), test_acc, bacc, f1, f1_macro, f1_micro))
    # print(confusion_matrix)
    return test_acc, bacc, f1, f1_macro, f1_micro


if __name__ == '__main__':
    start = time.time()
    args = parse_args()
            
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()

    seed_everything(args.seed)
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device = args.device)

    data = load_dataset(args, encoder)
    # print(get_unique_counts(data.y))
    edge_idxs = split_edge(data, args.train_ratio, args.val_ratio)

    res=[]

    train_edge0 = data.edge_index[:, edge_idxs['train']]
    
    train_val_edge = torch.cat([data.edge_index[:, edge_idxs['train']], data.edge_index[:, edge_idxs['val']]], dim = 1).to(args.device)
    train_edge = process_edge(train_edge0).to(args.device)
    train_edge, train_val_edge = train_edge.to(args.device), train_val_edge.to(args.device)
    train_edge0 = train_edge0.to(args.device)
    train_y = data.y[edge_idxs['train']]
    test_y = data.y[edge_idxs['test']]
    val_y = data.y[edge_idxs['val']]
    
    # ExtWF
    # num_classes = int(torch.unique(data.y).shape[0])    
    # num_nodes = data.num_nodes
    # k=1
    # top_k_similar = compute_similarity(data.edge_index[:, edge_idxs['train']], train_y, num_classes, num_nodes, k)
    # bac, f1 = edge_classification_and_metrics(
    #     data.edge_index[:, edge_idxs['train']], train_y, data.edge_index[:, edge_idxs['test']], test_y, top_k_similar, k, num_classes, num_nodes)
    # print(bac, f1)
    
    # _,_,args.entropy= cal_topo_reweight(data.edge_index[:, edge_idxs['train']], train_y.view(-1), args)
    # one_sim(data.edge_index[:, edge_idxs['train']], train_y.view(-1))
    # result, first, second = bimean_encoding(data, edge_idxs, train_y, 1000, args)
    # # densityplt(result)
    # res = mc(data, edge_idxs, train_y, 1000, args)
    # densityplt(res)
    # args.entropy= ge_reweight(data.edge_index[:, edge_idxs['train']], train_y.view(-1), first, second, args)
    # args.entropy= ge_reweight(data.edge_index[:, edge_idxs['train']], train_y.view(-1), args)
    
    # head, tail, args.entropy = cal_topo_reweight(data.edge_index[:, edge_idxs['train']], train_y.view(-1), args)
    # args.en1, args.en2, args.lab = ge_post(head, tail, 0.65)
    # el_ra = el_ratio(data.edge_index[:, edge_idxs['val']], val_y.view(-1))
    # # el_ra = el_ratio(data.edge_index[:, edge_idxs['train']], train_y.view(-1))
    # args.entropy, args.lab = quan_post(el_ra, 0.65)
    
    # # # # eh = edge_homophily(data.edge_index, data.y.view(-1))
    # # # # print(eh)
    # # # # args.entropy= cal_topo_reweight(data.edge_index, data.y.view(-1))
    # # # # Get topology imbalance ratio
    # # # # topo_imb_ratio(data)

    # # # # topology reweight
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
        optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=args.lr)
            
        # run(data, loaders, model, classifier, optimizer, loss_fn, train_edge0, train_edge, args)
        
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
        metrics_output.append("{}: {:.3f} $\pm$ {:.3f}".format(metric, means[metric], stds[metric]))
        
    output_string = f"Total time: {end-start:.2f} seconds," + ', '.join(metrics_output)

    # Print the output to the console
    print(output_string)
