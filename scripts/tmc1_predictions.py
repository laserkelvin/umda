
from joblib import load
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from umda import EmbeddingModel

embedder = load("../models/EmbeddingModel.pkl")
tmc1_df = load("../data/processed/tmc1_ready.pkl")
tmc1_df = tmc1_df.loc[tmc1_df["canonical"] != "[HH]"]
tmc1_df.reset_index(inplace=True, drop=True)

vectors = np.vstack([embedder(smi) for smi in tmc1_df["canonical"]])
form_index = tmc1_df.loc[tmc1_df["canonical"] == "C=O"].index.tolist().pop()

distances = np.sqrt(np.sum((vectors - vectors[form_index])**2., axis=1))
results = dict()

for grid in Path("../models").rglob("*_grid.pkl"):
    name = str(grid).split("_")[0].split("/")[-1]
    model = load(grid).best_estimator_
    tmc1_df[name] = model.predict(vectors)

print(tmc1_df)
tmc1_df["distance"] = distances
tmc1_df.to_csv("tmc1_results.csv", index=False)
