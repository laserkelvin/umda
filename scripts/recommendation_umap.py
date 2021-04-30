
import numpy as np
from umap import UMAP
from joblib import load
import h5py

embedding_model = load("../models/EmbeddingModel.pkl")
tmc1_data = load("../data/processed/tmc1_ready.pkl")
tmc1_data = tmc1_data.loc[tmc1_data["canonical"] != "[HH]"]
tmc1_data.reset_index(drop=True, inplace=True)
tmc1_smi = tmc1_data["canonical"].tolist()
vecs = np.vstack([embedding_model.vectorize(smi) for smi in tmc1_smi])
rec_vecs = list()
with open("targets.smi", "r") as read_file:
    for smi in read_file.readlines():
        rec_vecs.append(embedding_model.vectorize(smi))
rec_vecs = np.vstack(rec_vecs)
# stack them together
X = np.vstack([vecs, rec_vecs])

rng = np.random.RandomState(452190561)
umap_kwargs = {"n_neighbors": 100, "metric": "euclidean", "random_state": rng, "min_dist": 0.01, "spread": 1., "low_memory": False}

umap = UMAP(**umap_kwargs)
h5_data = h5py.File("../data/processed/recommendation_umap.h5", "a")
# fit the recommendation dataset
result = umap.fit_transform(X)
tmc1_mask = np.zeros(len(X), dtype=bool)
tmc1_mask[:len(vecs)] = True
for key in ["umap", "tmc1_mask"]:
    try:
        del h5_data[key]
    except:
        pass
h5_data["umap"] = result
h5_data["tmc1_mask"] = tmc1_mask

