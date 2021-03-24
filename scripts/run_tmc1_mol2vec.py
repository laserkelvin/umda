
import pandas as pd
import numpy as np
from umda import smi_vec

model = smi_vec.load_model("../models/mol2vec_model.pkl")

tmc1_df = pd.read_csv("../data/raw/TMC-1_inventory.csv")
tmc1_df = tmc1_df.loc[tmc1_df["Isotopologue"] == 0]

smiles = tmc1_df["SMILES"].to_list()
canonical_smi = [smi_vec.canonicize_smi(smi) for smi in smiles]
vectors = [smi_vec.smi_to_vector(smi, model, radius=1) for smi in canonical_smi]

tmc1_df["canonical"] = np.vstack(canonical_smi)
tmc1_df["vectors"] = vectors

tmc1_df.to_pickle("../data/processed/tmc1_ready.pkl")
