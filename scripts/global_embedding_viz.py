from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt

"""
This script is used to generate the UMAP visualization
of the dataset, as a function of each category of data
source. This takes the mol2vec embeddings, and attempts
to learn the projection from the PCA to 2 dimension space.
"""

logger.add("global_viz.log")

rng = np.random.RandomState(42)

umap_kwargs = {
    "n_neighbors": 100,
    "metric": "euclidean",
    "random_state": rng,
    }

embedding_path = "../data/processed/pipeline_embeddings_70.h5"

with h5py.File(embedding_path, "r") as h5_file:
    vectors = np.array(h5_file["pca"])
    logger.info(f"Shape of the vector matrix: {vectors.shape}")

logger.info("Initializing UMAP")
umap_model = UMAP(**umap_kwargs)

logger.info("Taking random sample of full dataset")
indices = np.arange(vectors.shape[0])
indices = rng.choice(indices, size=100000)
indices.sort()

chosen = vectors[indices]

logger.info("Fitting UMAP to chosen subset")
umap_model.fit(chosen)
# this is all the embeddings
umap_vecs = umap_model.transform(vectors)

with h5py.File("../data/processed/umap_vectors.h5", "a") as output:
    output["umap"] = umap_vecs
    output["chosen"] = chosen
    output["index"] = indices

