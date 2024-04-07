import pandas as pd
import pysmiles
import numpy as np
import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)
df = pd.read_parquet("test.parquet")
dict_atome_to_idx = {
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
mini = df.id.min()
maxi = df.id.max()
dict_protein_to_idx = {
    "BRD4":0,
    "HSA":1,
    "sEH":2,
}
def turn_to_graph(row):
    id = row.id
    if (id - mini)%(int((maxi-mini)/1000)) == 0:
        print("progress : ", (id - mini)//(int((maxi-mini)/1000)))
    smile = pysmiles.read_smiles(row.molecule_smiles)
    molecule = pd.DataFrame(dict(smile.nodes.data())).T
    molecule["element"] = molecule["element"].apply(lambda x: dict_atome_to_idx[x])
    molecule["aromatic"] = molecule["aromatic"].astype(int)
    nodes = molecule[["element","aromatic","charge","hcount"]].to_numpy(dtype = float)
    edges = []
    edges_attr = []
    for edge in smile.edges.data():
        edges.append(np.array([edge[0], edge[1]]))
        edges_attr.append(np.array(edge[2]['order']))
    edges = np.stack(edges).reshape(2,len(smile.edges()))
    edges_attr = np.stack(edges_attr).reshape(len(smile.edges()),-1)
    protein = np.array(dict_protein_to_idx[row.protein_name])
    return np.array(id), nodes, edges, edges_attr, protein

graphes = df.apply(turn_to_graph, axis = 1)
with open("graphes_test.npy", "wb") as f:
    np.save(f, graphes.to_numpy())