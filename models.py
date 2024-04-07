import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GCNConv, GATv2Conv
from torch.nn import Linear, ReLU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Model(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Model, self).__init__()
        n_emb_atom = 32
        n_emb_prot = 10
        self.atom_embedding = nn.Embedding(20, n_emb_atom)
        self.protein_embedding = nn.Embedding(3, n_emb_prot)
        self.graph_neural_net = Sequential('x, edge_index, edge_attr', [
                                (GCNConv(in_channels+n_emb_atom, 64), 'x, edge_index,edge_attr -> x'),
                                ReLU(inplace=True),
                                (GCNConv(64, 64), 'x, edge_index,edge_attr -> x'),
                                ReLU(inplace=True),
                                (GCNConv(64, 64), 'x, edge_index,edge_attr -> x'),
                                ReLU(inplace=True),
                                (GCNConv(64, 64), 'x, edge_index,edge_attr -> x'),
                                ReLU(inplace=True),
                                Linear(64, out_channels),
                            ])
        self.linear1 = nn.Linear(out_channels+n_emb_prot, 4*n_emb_atom)
        self.linear2 = nn.Linear(4*n_emb_atom, 3*n_emb_atom)
        self.linear3 = nn.Linear(3*n_emb_atom, n_emb_atom)
        self.linear4 = nn.Linear(n_emb_atom,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def extract_graph_features(self, data, y_hat):
        idx = data.batch.to(device)
        idx = idx.unsqueeze(1).repeat(1, y_hat.size(1))
        out = torch.zeros(torch.max(idx)+1, y_hat.size(1), device=device)
        out = out.scatter_add_(0, idx, y_hat)
        count = torch.zeros(torch.max(idx)+1, y_hat.size(1),device=device)
        count = count.scatter_add_(0, idx, torch.ones_like(y_hat,device=device))
        out = torch.div(out, count)
        return out
    def forward(self, data, protein):
        name_atoms = data.x[:,0].long()
        atom_embedding = self.atom_embedding(name_atoms)
        X = torch.cat([data.x[:,0:], atom_embedding], dim=1)
        output_graph = self.graph_neural_net(X, data.edge_index, data.edge_attr)
        output_protein = self.relu(self.protein_embedding(protein))
        output_graph = self.extract_graph_features(data, output_graph)
        input_linear = torch.cat([output_graph, output_protein], dim=1)
        output_linear = self.relu(self.linear1(input_linear))
        output_linear = self.relu(self.linear2(output_linear))
        output_linear = self.relu(self.linear3(output_linear))
        output_linear = self.sigmoid(self.linear4(output_linear))
        return output_linear

class Model_GAT(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Model, self).__init__()
        n_emb_atom = 32
        n_emb_prot = 10
        self.atom_embedding = nn.Embedding(20, n_emb_atom)
        self.protein_embedding = nn.Embedding(3, n_emb_prot)
        self.graph_neural_net = Sequential('x, edge_index, edge_attr', [
                                (GATv2Conv(in_channels+n_emb_atom, 64), 'x, edge_index,edge_attr -> x'),
                                ReLU(inplace=True),
                                (GATv2Conv(64, 128), 'x, edge_index,edge_attr -> x'),
                                ReLU(inplace=True),
                                (GATv2Conv(128, 128), 'x, edge_index,edge_attr -> x'),
                                ReLU(inplace=True),
                                (GATv2Conv(128, 64), 'x, edge_index,edge_attr -> x'),
                                ReLU(inplace=True),
                                Linear(64, out_channels),
                            ])
        self.linear1 = nn.Linear(out_channels+n_emb_prot, 4*n_emb_atom)
        self.linear2 = nn.Linear(4*n_emb_atom, 3*n_emb_atom)
        self.linear3 = nn.Linear(3*n_emb_atom, n_emb_atom)
        self.linear4 = nn.Linear(n_emb_atom,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def extract_graph_features(self, data, y_hat):
        idx = data.batch.to(device)
        idx = idx.unsqueeze(1).repeat(1, y_hat.size(1))
        out = torch.zeros(torch.max(idx)+1, y_hat.size(1), device=device)
        out = out.scatter_add_(0, idx, y_hat)
        count = torch.zeros(torch.max(idx)+1, y_hat.size(1),device=device)
        count = count.scatter_add_(0, idx, torch.ones_like(y_hat,device=device))
        out = torch.div(out, count)
        return out
    def forward(self, data, protein):
        name_atoms = data.x[:,0].long()
        atom_embedding = self.atom_embedding(name_atoms)
        X = torch.cat([data.x[:,0:], atom_embedding], dim=1)
        output_graph = self.graph_neural_net(X, data.edge_index, data.edge_attr)
        output_protein = self.relu(self.protein_embedding(protein))
        output_graph = self.extract_graph_features(data, output_graph)
        input_linear = torch.cat([output_graph, output_protein], dim=1)
        output_linear = self.relu(self.linear1(input_linear))
        output_linear = self.relu(self.linear2(output_linear))
        output_linear = self.relu(self.linear3(output_linear))
        output_linear = self.sigmoid(self.linear4(output_linear))
        return output_linear