from utils import batch_to_gpu, mixup, edge_emb
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
import time
import math


def train(data, model, classifier, loader, optimizer, loss_fn, train_edge0, prop_edge_index, args):

    model.train()
    classifier.train()
    losses = []

    for batch in loader:
        optimizer.zero_grad()
        idx1, idx2 = batch[0], batch[1]
        idx1, idx2 = idx1.to(args.device), idx2.to(args.device)
        
        x = model(prop_edge_index, args)
        edge_emb1 = edge_emb(x, data.edge_index, idx1)
        attr_emb = data.edge_attr[idx1]
        out = classifier(edge_emb1, attr_emb)
        
        loss0 = loss_fn(out, data.y[idx1].view(-1))
        
        if args.mixup !=0:
            edge_emb2, list, idx1_new, idx2_new, lam = mixup(x, data, idx1, idx2, train_edge0, args)
            attr2 = lam*data.edge_attr[idx1[list]]+(1-lam)*data.edge_attr[idx1_new]
            out2= classifier(edge_emb2, attr2)
            loss1 = lam*loss_fn(out2, data.y[idx1[list]].view(-1))
            loss2 = (1-lam)*loss_fn(out2, data.y[idx1_new].view(-1))

        if args.method == 'none':
            if args.mixup ==0:
                loss=loss0.mean()
            else:
                loss=loss0.mean()+args.h*(loss1+loss2).mean() 
        elif args.method == 'tw': 
            if args.mixup ==0:
                loss = (loss0 * args.topo_reweight[idx2]).mean()
            else:
                loss = (loss0 * args.topo_reweight[idx2]).mean()+args.h*(loss1*args.topo_reweight[idx2[list]]+loss2*args.topo_reweight[idx2_new]).mean()
        elif args.method == 'qw':
            loss = (loss0 * args.reweight[data.y[idx1].view(-1)]).mean()
            if args.mixup !=0:
                loss += args.h*(loss1*args.reweight[data.y[idx1[list].view(-1)]]+loss2*args.reweight[data.y[idx1_new].view(-1)]).mean()
        elif args.method == 'tq':
            loss = ( loss0 * args.w[idx2] ).mean()
            if args.mixup !=0:
                loss += args.h*(loss1*args.w[idx2[list]]+loss2*args.w[idx2_new]).mean()
                
        loss.backward()
        losses.append(loss.mean().item())
            
        optimizer.step()

    return np.mean(losses)

@torch.no_grad()
def eval(data, model, classifier, loader, prop_edge_index, args):
    model.eval()
    classifier.eval()

    y_true, y_pred = [], []
    for batch in loader:
        idx1, idx2 = batch[0], batch[1]
        idx1, idx2 = idx1.to(args.device), idx2.to(args.device)

        x = model(prop_edge_index, args)
        # out = classifier(x, data.edge_index, idx1, data.edge_attr)
        edge_emb1 = edge_emb(x, data.edge_index, idx1)
        attr_emb = data.edge_attr[idx1]
        out = classifier(edge_emb1, attr_emb)

        y_true.extend(data.y[idx1].view(-1).cpu().numpy().tolist())
        y_pred.extend(out.argmax(dim=1).cpu().numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, average = 'weighted')
    f1_macro = f1_score(y_true, y_pred, average = 'macro')
    f1_micro = f1_score(y_true, y_pred, average = 'micro')
    confusion = confusion_matrix(y_true, y_pred)
    
    return accuracy, bacc, w_f1, f1_macro, f1_micro, confusion, y_true, y_pred

@torch.no_grad()
def evaltrain(data, model, classifier, loader, prop_edge_index, args):
    model.eval()

    y_true, y_pred, y_pro, entropy= [], [], [], []
    for batch in loader:
        idx1, idx2 = batch[0], batch[1]
        idx1, idx2 = idx1.to(args.device), idx2.to(args.device)

        # out = model(data.edge_index, idx1, data.edge_attr, prop_edge_index)
        x = model(prop_edge_index, args)
        edge_emb1 = edge_emb(x, data.edge_index, idx1)
        attr_emb = data.edge_attr[idx1]
        out = classifier(edge_emb1, attr_emb)
        out = torch.softmax(out, dim=1)

        y_true.extend(data.y[idx1].view(-1).cpu().numpy().tolist())
        y_pred.extend(out.argmax(dim=1).cpu().numpy().tolist())
        y_pro.extend(out[:,0].cpu().numpy().tolist())
        idx2=idx2.cpu()
        entropy.extend(args.entropy[idx2].cpu().numpy().tolist())
        
    return y_true, y_pred, y_pro, entropy

@torch.no_grad()
def evaltrain(data, model, classifier, loader, prop_edge_index, args):
    model.eval()

    y_true, y_pred, y_pro, entropy= [], [], [], []
    for batch in loader:
        idx1, idx2 = batch[0], batch[1]
        idx1, idx2 = idx1.to(args.device), idx2.to(args.device)

        # out = model(data.edge_index, idx1, data.edge_attr, prop_edge_index)
        x = model(prop_edge_index, args)
        edge_emb1 = edge_emb(x, data.edge_index, idx1)
        attr_emb = data.edge_attr[idx1]
        out = classifier(edge_emb1, attr_emb)
        out = torch.softmax(out, dim=1)

        y_true.extend(data.y[idx1].view(-1).cpu().numpy().tolist())
        y_pred.extend(out.argmax(dim=1).cpu().numpy().tolist())
        y_pro.extend(out[:,0].cpu().numpy().tolist())
        idx2=idx2.cpu()
        entropy.extend(args.entropy[idx2].cpu().numpy().tolist())
        
    return y_true, y_pred, y_pro, entropy

@torch.no_grad()
def evaltrain2(data, model, classifier, loader, prop_edge_index, args):
    model.eval()
    
    y_true, y_pred, y_pro, all, label, entropy = [], [], [], [], [], []
    # en = (args.en1+args.en2)/2
    for batch in loader:
        idx1, idx2 = batch[0], batch[1]
        idx1, idx2 = idx1.to(args.device), idx2.to(args.device)

        # out = model(data.edge_index, idx1, data.edge_attr, prop_edge_index)
        x = model(prop_edge_index, args)
        edge_emb1 = edge_emb(x, data.edge_index, idx1)
        attr_emb = data.edge_attr[idx1]
        out = classifier(edge_emb1, attr_emb)
        out = torch.softmax(out, dim=1)

        y_true.extend(data.y[idx1].view(-1).cpu().numpy().tolist())
        y_pred.extend(out.argmax(dim=1).cpu().numpy().tolist())
        y_pro.extend(out[:,0].cpu().numpy().tolist())
        idx2=idx2.cpu()
        # all.extend(en[idx2].cpu().numpy().tolist())
        label.extend(args.lab[idx2].cpu().numpy().tolist())
        entropy.extend(args.entropy[idx2].cpu().numpy().tolist())
    return y_true, y_pred, y_pro, all, label, entropy