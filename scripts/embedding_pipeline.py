
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
from sklearn.pipeline import make_pipeline
from loguru import logger
from joblib import dump


logger.add("embedding.log")

# for stochastic reproducibility
seed = 42
rng = np.random.default_rng(seed)

logger.info("Loading mol2vec model.")

m2v_model = smi_vec.load_model("../models/mol2vec_model.pkl")

# number of feature dimensions for the embedding model
embedding_dim = 300
pca_dim = 100
n_clusters = 20
h5_file = h5py.File(f"../data/processed/smiles_embeddings_{embedding_dim}.h5", "a")


RERUN = False

logger.info(f"mol2vec embedding dimension size: {embedding_dim}")
logger.info(f"PCA reduced dimensionality size: {pca_dim}")
logger.info(f"Will perform vectorization? {RERUN}")

if RERUN:
    logger.info("Reading in list of SMILES")

    with open("../data/interim/collected_smiles.smi") as read_file:
        smi_list = read_file.readlines()

    smi_list = list(map(lambda x: x.strip(), smi_list))

    logger.info("Beginning vectorization of SMILES.")
    smi_vec.serial_smi_vectorization(smi_list, m2v_model, h5_file, embedding_dim)
else:
    del h5_file["pca"]
    del h5_file["cluster_ids"]

logger.info("Beginning PCA dimensionality reduction")
# perform PCA dimensionality reduction
pca_model = IncrementalPCA(n_components=pca_dim)
vectors = h5_file["vectors"]
pca_model.fit(vectors)
h5_file["pca"] = pca_model.transform(vectors)

logger.info("Saving the trained PCA model.")
dump(pca_model, "../models/pca_model.pkl")

logger.info("Performing K-means clustering on dataset")
kmeans = KMeans(n_clusters=n_clusters, random_state=seed, metric="cosine")
kmeans.fit(h5_file["pca"])
h5_file["cluster_ids"] = kmeans.labels_
dump(kmeans, "../models/kmeans_model.pkl")

logger.info("Combining the models into a pipeline")
pipe = make_pipeline(pca_model, kmeans)
dump(pipe, "../models/embedding_pipeline.pkl")

h5_file.close()