
import numpy as np
import pandas as pd
from umap import UMAP
import h5py

from umda.data import load_pipeline, load_data
from umda.utils import paths

"""
This script trains a UMAP model on the combined
context of TMC-1 molecules and new recommendations.

This will look for a CSV file that has cleaned up
the recommendations in the `notebooks/reports` folder
for the SMILES to process. The resulting projected
UMAP data is saved to an HDF5 file in the `processed`
folder.
"""

embedding_model = load_pipeline()
tmc1_data = load_data()[-1]
vecs = np.vstack([embedding_model.vectorize(smi) for smi in tmc1_data["SMILES"]])
rec_vecs = list()

recs = pd.read_csv("../notebooks/reports/tmc1_recommendations_latest.csv")
rec_vecs = np.vstack([embedding_model(smi) for smi in recs["Recommendation"]])
# stack them together
X = np.vstack([vecs, rec_vecs])

rng = np.random.RandomState(452190561)
umap_kwargs = {"n_neighbors": 100, "metric": "euclidean", "random_state": rng, "min_dist": 0.01, "spread": 1., "low_memory": False}

umap = UMAP(**umap_kwargs)
h5_data = h5py.File(paths.get("processed").joinpath("recommendation_umap.h5"), "a")
# fit the recommendation dataset
result = umap.fit_transform(X)
tmc1_mask = np.zeros(len(X), dtype=bool)
tmc1_mask[:len(vecs)] = True
# if the keys exist, delete them and overwrite
for key in ["umap", "tmc1_mask"]:
    try:
        del h5_data[key]
    except:
        pass
h5_data["umap"] = result
h5_data["tmc1_mask"] = tmc1_mask

