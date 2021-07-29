
from joblib import load
from pathlib import Path
from sklearn.metrics import pairwise_distances
import h5py
import numpy as np
import pandas as pd

from umda.data import load_pipeline, load_data

embedder = load_pipeline()
tmc1_df = load_data()[-1]

# do linear interpolation between the two molecules, and find
# the molecules in TMC-1 closest to the interpolated values
begin = embedder.vectorize("C=O")
end = embedder.vectorize("N#CC1=CC=CC2=CC=CC=C12")

diff = (end - begin)[None,:]
x = np.linspace(0., 1., 8)[:,None]
values = begin + (diff * x)

tmc1_vecs = np.vstack([embedder.vectorize(smi) for smi in tmc1_df["SMILES"]])
dists = pairwise_distances(values, tmc1_vecs)
chosen = list()
for index, dist in enumerate(dists):
    cand_indices = np.argsort(dist)
    for cand_idx in cand_indices:
        if cand_idx not in chosen:
            chosen.append(cand_idx)
            break
select = tmc1_df.iloc[chosen]
print(select["Molecule"])
targets = select["SMILES"].tolist()
print(targets)

columns = select["Column density (cm^-2)"].values
vectors = np.vstack([embedder(smi) for smi in targets])

distances = np.sqrt(np.sum((vectors - vectors[0])**2., axis=1))
results = dict()
best_models = load("../notebooks/estimator_training/outputs/grid_search/best_models.pkl")

for name, model in best_models.items():
    results[name] = model.predict(vectors)

df = pd.DataFrame(results)
df["smiles"] = targets
df["distance"] = distances
df["truth"] = np.log10(columns)
df.sort_values("distance", inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv("demoset_results.csv", index=False)
