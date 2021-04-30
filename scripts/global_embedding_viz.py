from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

"""
This script is used to generate the UMAP visualization
of the dataset, as a function of each category of data
source. This takes the mol2vec embeddings, and attempts
to learn the projection from the PCA to 2 dimension space.
"""

logger.add("global_viz.log")

rng = np.random.RandomState(42)

umap_kwargs = {
    "n_neighbors": 50,
    "metric": "euclidean",
    "random_state": rng,
    "min_dist": 0.01,
    "spread": 1.,
    "low_memory": False
    }

embedding_path = "../data/processed/pipeline_embeddings_70.h5"

with h5py.File(embedding_path, "r") as h5_file:
    vectors = np.array(h5_file["pca"]).astype(np.float32)
    logger.info(f"Shape of the vector matrix: {vectors.shape}")

logger.info("Initializing UMAP")
umap_model = UMAP(**umap_kwargs)

logger.info("Taking random sample of full dataset")
indices = np.arange(vectors.shape[0])
indices = rng.choice(indices, size=100000)
indices.sort()

chosen = np.ascontiguousarray(vectors[indices]).astype(np.float32)

logger.info("Fitting UMAP to chosen subset")
umap_model.fit(chosen)
# this is all the embeddings
combined = list()
# chunk the projection as we don't have enough memory :P
logger.info("Fit done. Now projecting the full dataset iteratively.")
for index, chunk in tqdm(enumerate(np.array_split(vectors, 50))):
    logger.info(f"Chunk {index+1}, chunk size {chunk.shape}")
    combined.append(umap_model.transform(chunk))
umap_vecs = np.vstack(combined)

with h5py.File("../data/processed/umap_vectors.h5", "a") as output:
    for key in ["umap", "chosen", "index"]:
        try:
            del output[key]
        except KeyError:
            pass
    output["umap"] = umap_vecs
    output["chosen"] = chosen
    output["index"] = indices

