
from joblib import load
import numpy as np
import pandas as pd

from umda.data import load_data, load_pipeline

embedder = load_pipeline()
tmc1_df = load_data()[-1]

# generate the molecule embeddings
vectors = np.vstack([embedder(smi) for smi in tmc1_df["SMILES"]])
form_index = tmc1_df.loc[tmc1_df["canonical"] == "C=O"].index.tolist().pop()

distances = np.sqrt(np.sum((vectors - vectors[form_index])**2., axis=1))
results = dict()

# load in the trained models
best_models = load("../notebooks/estimator_training/outputs/grid_search/best_models.pkl")
for name, model in best_models.items():
    tmc1_df[name] = model.predict(vectors)

tmc1_df["distance"] = distances
tmc1_df.to_csv("tmc1_results.csv", index=False)
