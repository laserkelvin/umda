
import pandas as pd
import h5py
from joblib import load

from umda import paths


def load_data(exclude_hydrogen: bool=True):
    data_path = paths.get("processed").joinpath("pipeline_embeddings_70.h5")
    tmc1_data = pd.read_pickle(paths.get("processed").joinpath("tmc1_ready.pkl"))
    if exclude_hydrogen:
        tmc1_data = tmc1_data.loc[tmc1_data["SMILES"] != "[HH]"]
    tmc1_data.reset_index(inplace=True, drop=True)
    with h5py.File(data_path, "r") as h5_data:
        X = h5_data["pca"][:]
        cluster_ids = h5_data["cluster_ids"][:]
    return X, cluster_ids, tmc1_data


def load_pipeline():
    model_path = paths.get("models")
    embedder = load(model_path.joinpath("EmbeddingModel.pkl"))
    return embedder