"""
Embedding pipeline

This script will take a collected list of SMILES, and generate all of the
vector embeddings and perform transformations to prepare it for analysis.

Because we're dealing with potentially large datasets, it's important to
be mindful of the amount of memory you have access to, particularly for the
K-means step! If you have memory issues, I suggest changing over to the
dask_ml versions of sklearn algorithms for this step.
"""

USE_DASK = False

import h5py
import numpy as np
import pandas as pd
from joblib import parallel_backend
from umda import smi_vec, EmbeddingModel
from dask import array as da
from dask.distributed import Client, LocalCluster

if USE_DASK:
    from dask_ml.decomposition import IncrementalPCA
    from dask_ml.cluster import KMeans
    from dask_ml.preprocessing import StandardScaler
else:
    from sklearn.decomposition import IncrementalPCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
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
pca_dim = 70
n_clusters = 20
n_workers = 8
h5_target = f"../data/processed/smiles_embeddings_{embedding_dim}.h5"
output_target = f"../data/processed/pipeline_embeddings_{pca_dim}.h5"
# h5_file = h5py.File(f"../data/processed/smiles_embeddings_{embedding_dim}.h5", "a")

def train_fit_model(data: np.ndarray, model, dask: bool = False, n_jobs: int = 8):
    """
    This function just helps simplify the main code by handling various contexts.
    If `dask` is being used, we use the dask backend for computation as well
    as making sure that the result is actually computed.
    """
    if dask:
        backend = "dask"
    else:
        backend = "threading"
    with parallel_backend(backend, n_jobs):
        model.fit(data)
        transform = model.transform(data)
        if dask:
            transform = transform.compute()
        # if we are fitting a clustering model we grab the labels
        labels = getattr(model, "labels_", None)
        if dask and labels is not None:
            labels = labels.compute()
    return (model, transform, labels)


RERUN = True

logger.info(f"mol2vec embedding dimension size: {embedding_dim}")
logger.info(f"PCA reduced dimensionality size: {pca_dim}")
logger.info(f"Will perform vectorization? {RERUN}")

if RERUN:
    logger.info("Reading in list of SMILES")

    df = pd.read_pickle("../data/processed/combined_smiles.pkl.bz2")
    smi_list = df["Raw"].tolist()

    logger.info("Beginning vectorization of SMILES.")
    with h5py.File(h5_target, "a") as embeddings_file:
        for key in ["labels", "smiles", "vectors"]:
            try:
                del embeddings_file[key]
            except KeyError:
                pass
        smi_vec.serial_smi_vectorization(smi_list, m2v_model, embeddings_file, vec_length=embedding_dim)
        embeddings_file.create_dataset("labels", data=df["Labels"].values)


embeddings_file = h5py.File(h5_target, "r")
output_file = h5py.File(output_target, "a")

if USE_DASK:
    client = Client(threads_per_worker=2, n_workers=8)
    vectors = da.from_array(embeddings_file["vectors"])
else:
    vectors = embeddings_file["vectors"][:]


scaler = StandardScaler()
pca_model = IncrementalPCA(n_components=pca_dim)
kmeans = KMeans(n_clusters=n_clusters, random_state=seed)

# preprocess the embeddings
vectors = scaler.fit_transform(vectors)

logger.info("Beginning PCA dimensionality reduction")
# perform PCA dimensionality reduction
pca_model = IncrementalPCA(n_components=pca_dim)
pca_model, transformed, _ = train_fit_model(vectors, pca_model, USE_DASK, n_workers)
# save both the reduced dimension vector and the full
output_file["pca"] = transformed
output_file["explained_variance"] = pca_model.explained_variance_ratio_

logger.info("Saving the trained PCA model.")
dump(pca_model, "../models/pca_model.pkl")

logger.info("Performing K-means clustering on dataset")
kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
kmeans, _, labels = train_fit_model(output_file["pca"], kmeans, USE_DASK, n_workers)
output_file["cluster_ids"] = labels
dump(kmeans, "../models/kmeans_model.pkl")

logger.info("Combining the models into a pipeline")
pipe = make_pipeline(scaler, pca_model, kmeans)
dump(pipe, "../models/embedding_pipeline.pkl")

output_file.close()
embeddings_file.close()

# generate a convenient wrapper for all the functionality
embedder = EmbeddingModel(m2v_model, transform=pipe)
dump(embedder, "../models/EmbeddingModel.pkl")
