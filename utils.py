import torch
import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)
import pysmiles
import networkx as nx
from torch_geometric.data import  Data, Batch, Dataset
import pandas as pd
from torch.utils.data import DataLoader
class dataset(Dataset):
    def __init__(self, root):
        super(dataset, self).__init__()
        self.root = root
        self.df = pd.read_parquet(root)
        self.dict_protein_to_idx = {
            "BRD4":0,
            "HSA":1,
            "sEH":2,
        }
        self.dict_atome_to_idx = {
                "C":0,
                "H":1,
                "O":2,
                "N":3,
                "Dy":4,
                "Br":5,
                "Cl":6,
                "F":7,
                "I":8,
                "P":9,
                "S":10,
                "B":11,
                "Si":12,
                "Se":13,
                "As":14,
                "Ge":15,
                "Sn":16,
            }
        self.dico_graphe = {}
        self.dico_protein = {}
        self.preprocess()
    def preprocess(self):
        for idx, row in self.df.iterrows():
            protein = torch.tensor(self.dict_protein_to_idx[row.protein_name])
            smile = pysmiles.read_smiles(row.molecule_smiles)
            nodes = []
            for _, node in enumerate(smile.nodes.data()):
                node = node[1]
                nodes.append(torch.tensor([self.dict_atome_to_idx[node['element']],
                                        float(node['aromatic']),
                                            float(node['charge']), 
                                            float(node['hcount'])]))
            nodes = torch.stack(nodes).reshape(len(smile.nodes()),-1)
            edges = []
            edges_attr = []
            for edge in smile.edges.data():
                edges.append(torch.tensor([edge[0], edge[1]]))
                edges_attr.append(torch.tensor(edge[2]['order']))
            edges = torch.stack(edges).reshape(2,len(smile.edges()))
            edges_attr = torch.stack(edges_attr).reshape(len(smile.edges()),-1)
            y = torch.tensor(row.binds)
            data = Data(x=nodes, edge_index=edges, edge_attr=edges_attr, y=y)
            self.dico_graphe[idx] = data
            self.dico_protein[idx] = protein
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self,idx):
        return self.dico_graphe[idx], self.dico_protein[idx]

def collate_fn(batch):
    data = [x[0] for x in batch]
    protein = [x[1] for x in batch]
    graph = Batch.from_data_list(data)
    return graph, torch.stack(protein)