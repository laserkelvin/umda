import h5py
import numpy as np
import pandas as pd
from periodictable import elements
from dask import array as da
from dask_ml.metrics import pairwise_distances
from joblib import load
from tqdm.auto import tqdm

# we just want to recommend molecules with these elements
keep_elements = ["H", "C", "O", "N", "P", "S", "Si"]
element_filter = [el.symbol for el in elements if el.symbol not in keep_elements]


def is_not_pure_hydrogen(smi: str) -> bool:
    # this function checks if the molecule only contains
    # hydrogen, which stuffs up the obabel geometry generation
    characters = list(set([c for c in smi if c.isalpha()]))
    if (len(characters) == 1) and (characters[0] == "H"):
        return False
    return True


def is_not_complex(smi: str) -> bool:
    return "." not in smi


embedding_model = load("../models/EmbeddingModel.pkl")
tmc1_df = load("../data/processed/tmc1_ready.pkl")
# let's ignore H2
tmc1_df = tmc1_df.loc[tmc1_df["canonical"] != "[HH]"]
tmc1_df.reset_index(drop=True, inplace=True)
# generate the embeddings for the TMC-1 molecules
tmc1_smi = tmc1_df["canonical"].tolist()
vecs = np.vstack([embedding_model.vectorize(smi) for smi in tmc1_smi])
# load the full dataset ready to go
precomputed_h5 = h5py.File("../data/processed/pipeline_embeddings_70.h5", "r")
full_dataset = h5py.File("../data/processed/smiles_embeddings_300.h5", "r")
dataset = da.from_array(precomputed_h5["pca"])
full_smiles = np.array(full_dataset["smiles"])
# get the GP regressor
gp_model = load("../notebooks/estimator_training/outputs/grid_search/best_models.pkl")["gpr"]

results = list()
for tmc_index, vector in tqdm(enumerate(vecs), total=len(tmc1_smi)):
    # we don't really care about the distance values, just the index
    distances = pairwise_distances(dataset, vector[None, :]).ravel().compute()
    sorted_indices = np.argsort(distances)
    top_ten = sorted_indices[:100]
    for index in top_ten:
        # ignore if it's the same SMILES code
        if tmc1_smi[tmc_index] != full_smiles[index]:
            results.append(
                {
                    "entry": index,
                    "recommendation": full_smiles[index],
                    "anchor": tmc1_smi[tmc_index],
                    "distance": distances[index],
                }
            )

rec_df = pd.DataFrame(results)
# first filter out non-unique suggests by their database index
rec_df.drop_duplicates("entry", inplace=True)
# now filter out molecules that contain elements that are not in our keep list
rec_df = rec_df.loc[
    (~rec_df["recommendation"].apply(lambda x: any([el in x for el in element_filter]))) &
    (rec_df["recommendation"].apply(is_not_complex)) & # removes complexes
    (rec_df["recommendation"].apply(is_not_pure_hydrogen)) # removes pure hydrogen
]
rec_df.reset_index(inplace=True, drop=True)
# now we run predictions for the column densities
rec_vecs = np.vstack([embedding_model.vectorize(smi) for smi in rec_df["recommendation"]])
columns, std = gp_model.predict(rec_vecs, return_std=True)
rec_df["gpr_column"] = columns
rec_df["uncertainty"] = std
rec_df.to_csv("tmc1_recommendations.csv", index=False)
# now write out a smiles file
with open("targets.smi", "w+") as write_file:
    write_file.write("\n".join(rec_df["recommendation"].tolist()))

# do the complete set including TMC-1 molecules
full = np.vstack([vecs, rec_vecs])
_, cov = gp_model.predict(full, return_cov=True)
np.save("tmc1_recommendations_cov", cov)

