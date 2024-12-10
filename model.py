import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv
    
class GCNEncoder(nn.Module):
    def __init__(self, n_nodes, n_embed, n_hidden):
        super(GCNEncoder, self).__init__()
        
        self.embedding = nn.Embedding(n_nodes, n_embed)
        self.conv1 = GCNConv(n_embed, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, prop_edge_index, args):
        x, edge_index = self.embedding.weight, prop_edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=args.training)
        x = self.conv2(x, edge_index)
        return x
    
class GATEncoder(nn.Module):
    def __init__(self, n_nodes, n_embed, n_hidden):
        super(GATEncoder, self).__init__()
        
        self.embedding = nn.Embedding(n_nodes, n_embed)
        self.conv1 = GATConv(n_embed, n_hidden)
        self.conv2 = GATConv(n_hidden, n_hidden)
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    
    def forward(self, prop_edge_index, args):
        x, edge_index = self.embedding.weight, prop_edge_index
        x = F.dropout(x, p=args.dropout, training=args.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=args.training)
        x = self.conv2(x, edge_index)
        return x
    
class SAGEEncoder(nn.Module):
    def __init__(self, n_nodes, n_embed, n_hidden):
        super(SAGEEncoder, self).__init__()
        
        self.embedding = nn.Embedding(n_nodes, n_embed)
        self.conv1 = SAGEConv(n_embed, n_hidden)
        self.conv2 = SAGEConv(n_hidden, n_hidden)
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, prop_edge_index, args):
        x, edge_index = self.embedding.weight, prop_edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=args.training)
        x = self.conv2(x, edge_index)
        return x
    
class ChebEncoder(nn.Module):
    def __init__(self, n_nodes, n_embed, n_hidden):
        super(ChebEncoder, self).__init__()
        
        self.embedding = nn.Embedding(n_nodes, n_embed)
        self.conv1 = ChebConv(n_embed, n_hidden, K = 1)
        self.conv2 = ChebConv(n_hidden, n_hidden, K = 1)
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.conv1.reset_parameters()
        
    def forward(self, prop_edge_index, args):
        x = self.conv1(self.embedding.weight, prop_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=args.dropout, training=args.training)
        x = self.conv2(x, prop_edge_index)
        return x

# class Classifier(nn.Module):
#     def __init__(self, n_hidden, n_edge_attr, n_out, dropout):
#         super(Classifier, self).__init__()
        
#         self.edge_attr_lin = nn.Linear(n_edge_attr, n_hidden)
#         self.classifier = nn.Linear(n_hidden*2 + n_hidden, n_out)
        
#         self.dropout = nn.Dropout(dropout)
    
#     def reset_parameters(self):
#         self.edge_attr_lin.reset_parameters()
#         self.classifier.reset_parameters()
        
#     def forward(self, emb, attr):
#         edge_attr_emb = self.edge_attr_lin(attr)
        
#         edge_emb = torch.cat([emb, edge_attr_emb], dim=1)

#         return self.classifier(edge_emb)
    
class Classifier(nn.Module):
    def __init__(self, n_hidden, n_edge_attr, n_out, dropout):
        super(Classifier, self).__init__()
        self.edge_attr_lin = nn.Linear(n_edge_attr, n_hidden)
        self.relu = nn.ReLU()
        self.classifier_1 = nn.Linear(n_hidden*2 + n_hidden, n_hidden)
        self.classifier_2 = nn.Linear(n_hidden, n_out)
        self.dropout = nn.Dropout(dropout)
    
    def reset_parameters(self):
        self.edge_attr_lin.reset_parameters()
        self.classifier_1.reset_parameters()
        self.classifier_2.reset_parameters()
        
    def forward(self, emb, attr):
        edge_attr_emb = self.relu(self.edge_attr_lin(attr))
        edge_emb = torch.cat([emb, edge_attr_emb], dim=1)
        edge_emb = self.dropout(F.relu(self.classifier_1(edge_emb)))
        
        return self.classifier_2(edge_emb)