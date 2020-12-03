from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from umda.smi_vec import inchi_to_smiles, canonicize_smi
from loguru import logger
from joblib import Parallel, delayed

"""
pool_smiles.py

This script will combine the SMILES strings from multiple datasets
and pool them into a single dataframe/file ready to be used by
mol2vec.
"""


logger.add("smiles_concatenation.log")

# grab QM9
qm9 = pd.read_csv("../data/external/gdb9_prop_smiles.csv.tar.gz")
smi_list = qm9["smiles"].dropna().to_list()
logger.info(f"Dataset size with QM9: {len(smi_list)}")
# cleanup
del qm9

# getting the ZINC set
for smi_file in Path("../data/external").rglob("*/*.smi"):
    temp = smi_file.read_text().split("\n")[1:]
    for line in temp:
        if len(line) != 0:
            smi_list.append(line.split(" ")[0])

logger.info(f"Dataset size with ZINC: {len(smi_list)}")

# getting the PCBA set (biology)
pcba = pd.read_csv("../data/external/pcba.csv")
smi_list.extend(pcba["smiles"].dropna().to_list())
del pcba

logger.info(f"Dataset size with PCBA: {len(smi_list)}")

kida_df = pd.read_csv("../data/external/kida-molecules_05_Jul_2020.csv")
kida_df["SMILES"] = kida_df["InChI"].apply(inchi_to_smiles)

# Extract only those with SMILES strings
kida_smiles = kida_df.loc[(kida_df["SMILES"].str.len() != 0.)].dropna()
# append all the KIDA entries to our full list
smi_list.extend(kida_smiles["SMILES"].to_list())

del kida_df

logger.info(f"Dataset size with KIDA: {len(smi_list)}")

tmc1 = pd.read_csv("../data/raw/TMC-1_inventory.csv")
tmc1 = tmc1.loc[(tmc1["Isotopologue"] == 0)]
missing = tmc1.loc[~tmc1["SMILES"].isin(smi_list), "SMILES"].to_list()
smi_list.extend(missing)

logger.info(f"Dataset size with TMC-1: {len(smi_list)}")

# Add some PAHs by hand
with open("../data/external/hand_pah.smi") as read_file:
    pah_smi = read_file.read().split()
    smi_list.extend(pah_smi)

logger.info(f"Dataset size with PAHs: {len(smi_list)}")

logger.info("Canonicizing all SMILES.")
smi_list = Parallel(n_jobs=6)(delayed(canonicize_smi)(smi) for smi in smi_list)
# smi_list = [canonicize_smi(smi) for smi in smi_list]

final_df = pd.DataFrame(smi_list, columns=["Raw"])
# calculating molecular weights
molecular_weights = list()
for smi in smi_list:
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    molecular_weights.append(Descriptors.ExactMolWt(mol))
final_df["MW"] = molecular_weights

# remove duplicate entries according to canonical SMILES
final_df.drop_duplicates("Raw", inplace=True)
final_df.reset_index(inplace=True, drop=True)

logger.info(f"Final size of dataset without duplication: {len(final_df)}.")

final_df.to_pickle("../data/processed/combined_smiles.pkl.bz2")
# save a separate SMILES file for mol2vec use
final_df["Raw"].to_csv("../data/interim/collected_smiles.smi", sep="\n", index=False, header=None)