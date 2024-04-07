import torch
import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)
import pysmiles
import networkx as nx
from torch_geometric.data import  Data, Batch, Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
class dataset(Dataset):
    def __init__(self, root):
        super(dataset, self).__init__()
        self.root = root
        self.data = np.load(self.root, allow_pickle = True)
        self.data = np.random.permutation(self.data)
    def __len__(self):
        return self.data.shape[0]//5

    def __getitem__(self,idx):
        id, nodes, edges, edges_attr, protein,y = self.data[idx]
        id = torch.tensor(id)
        nodes = torch.from_numpy(nodes).float()
        edges = torch.from_numpy(edges).long()
        edges_attr = torch.from_numpy(edges_attr).float()
        protein = torch.tensor(protein)
        y = torch.tensor(y).float()
        data = Data(x=nodes, edge_index=edges, edge_attr=edges_attr, y=y)
        return data, protein, y



def collate_fn(batch):
    data = [x[0] for x in batch]
    protein = [x[1] for x in batch]
    graph = Batch.from_data_list(data)
    return graph, torch.stack(protein)


class submission_dataset(Dataset):
    def __init__(self, root):
        super(submission_dataset, self).__init__()
        self.root = root
        self.data = np.load(self.root, allow_pickle = True)
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        id, nodes, edges, edges_attr, protein = self.data[idx]
        id = torch.tensor(id)
        nodes = torch.from_numpy(nodes).float()
        edges = torch.from_numpy(edges).long()
        edges_attr = torch.from_numpy(edges_attr).float()
        protein = torch.tensor(protein)
        data = Data(x=nodes, edge_index=edges, edge_attr=edges_attr)
        return data, protein, id
    
def collate_fn_sub(batch):
    data = [x[0] for x in batch]
    protein = [x[1] for x in batch]
    idx = [x[2] for x in batch]
    graph = Batch.from_data_list(data)
    return graph, torch.stack(protein), torch.stack(idx)

