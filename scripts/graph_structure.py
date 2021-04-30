import h5py
import numpy as np
import networkx as nx
from joblib import load
from tqdm.auto import tqdm

graph = nx.Graph()

"""
Layout of the graph data structure for this project

Nodes are molecules, xy determined by UMAP
Node properties include x, y, SMILES identifier, is_tmc, column density
Edges can correspond to rate coefficients
"""

with h5py.File("../data/processed/umap_vectors.h5", "r") as data:
    umap_data = data["umap"][:]

combined_data = load("../data/processed/combined_smiles.pkl.bz2")
combined_data["x"] = umap_data[:,0]
combined_data["y"] = umap_data[:,1]

# despite being called Raw, the column are canonical SMILES
for index, row in tqdm(combined_data.iterrows()):
    graph.add_node(index, x=row["x"], y=row["y"], smi=row["Raw"], logncol=0., source=row["Labels"])

nx.write_gpickle(graph, "../data/processed/umda_graph.gpkl.gz")
