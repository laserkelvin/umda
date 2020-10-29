
"""
Embedding pipeline

This script will take a collected list of SMILES, and generate all of the
vector embeddings and perform transformations to prepare it for analysis.

Because we're dealing with potentially large datasets, it's important to
be mindful of the amount of memory you have access to, particularly for the
K-means step! If you have memory issues, I suggest changing over to the
dask_ml versions of sklearn algorithms for this step.
"""

import h5py
import numpy as np
from umda import smi_vec, EmbeddingModel
# from dask_ml.decomposition import IncrementalPCA
# from dask_ml.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans


# for stochastic reproducibility
rng = np.random.default_rng(42)

m2v_model = smi_vec.load_model("../models/mol2vec_model.pkl")

# number of feature dimensions for the embedding model
embedding_dim = 300

with open("../data/interim/collected_smiles.smi") as read_file:
    smi_list = read_file.readlines()

smi_list = list(map(lambda x: x.strip(), smi_list))

h5_file = h5py.File(f"../../data/processed/smiles_embeddings_{embedding_dim}.h5", "a")
smi_vec.serial_smi_vectorization(smi_list, m2v_model, h5_file, embedding_dim)

# perform PCA dimensionality reduction
pca_model = IncrementalPCA()