from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from umda.smi_vec import inchi_to_smiles, canonicize_smi
from loguru import logger
from joblib import Parallel, delayed
from tqdm.auto import tqdm

"""
pool_smiles.py

This script will combine the SMILES strings from multiple datasets
and pool them into a single dataframe/file ready to be used by
mol2vec.
"""


logger.add("smiles_concatenation.log")

# this is for storing which molecule comes from which database
labels = list()

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
            labels.append(0)

logger.info(f"Dataset size with ZINC: {len(smi_list)}")

# getting the PCBA set (biology)
pcba = pd.read_csv("../data/external/pcba.csv")
pcba_smi = pcba["smiles"].dropna().to_list()
smi_list.extend(pcba_smi)
labels.extend([1 for _ in range(len(pcba_smi))])
del pcba

logger.info(f"Dataset size with PCBA: {len(smi_list)}")

kida_df = pd.read_csv("../data/external/kida-molecules_05_Jul_2020.csv")
kida_df["SMILES"] = kida_df["InChI"].apply(inchi_to_smiles)

# Extract only those with SMILES strings
kida_smiles = kida_df.loc[(kida_df["SMILES"].str.len() != 0.0)].dropna()["SMILES"].to_list()
# append all the KIDA entries to our full list
smi_list.extend(kida_smiles)
labels.extend([2 for _ in range(len(kida_smiles))])

del kida_df

logger.info(f"Dataset size with KIDA: {len(smi_list)}")

tmc1 = pd.read_csv("../data/raw/TMC-1_inventory.csv")
tmc1 = tmc1.loc[(tmc1["Isotopologue"] == 0)]
tmc1_smi = tmc1["SMILES"].to_list()
#missing = tmc1.loc[~tmc1["SMILES"].isin(smi_list), "SMILES"].to_list()
smi_list.extend(tmc1_smi)
labels.extend([3 for _ in range(len(tmc1_smi))])

logger.info(f"Dataset size with TMC-1: {len(smi_list)}")

# Add some PAHs by hand
with open("../data/external/hand_pah.smi") as read_file:
    pah_smi = read_file.read().split()
    smi_list.extend(pah_smi)
    labels.extend([4 for _ in range(len(pah_smi))])

logger.info(f"Dataset size with hand picked PAHs: {len(smi_list)}")

# Add in the NASA IR spectral database
with open("../data/external/nasa_pah.smi") as read_file:
    pah_smi = read_file.read().split()
    smi_list.extend(pah_smi)
    labels.extend([5 for _ in range(len(pah_smi))])

logger.info(f"Dataset size with NASA PAHs: {len(smi_list)}")

# Add in the Yalamanchi et al. 2020 dataset
with open("../data/external/yalamanchi_2020.smi") as read_file:
    pah_smi = read_file.read().split()
    smi_list.extend(pah_smi)
    labels.extend([6 for _ in range(len(pah_smi))])

logger.info(f"Dataset size with small cyclic hydrocarbons: {len(smi_list)}")

logger.info("Adding PubChem canonical SMILES")
# add the PubChem pre-sanitized
with open("../data/external/pubchem/pubchem_screened.smi", "r") as read_file:
    pubchem_smi = read_file.readlines()
    smi_list.extend(pubchem_smi)
    labels.extend([7 for _ in range(len(pubchem_smi))])

logger.info(f"Dataset size with PubChem: {len(smi_list)}")

# exclusion list of elements
exclude_list = [
    "P", "Ho", "As", "Zr", "Sn", "V", "Au", "Br", "F", "Re", "U", "Al", "In", "Cl", "Ag", "Ce", "Hg", "Ta", "Te", "Ni", "Be", "Li", "Sb", "Bi", "Mn", "Co", "Tm", "Si",
    "Mo", "Se"
]

logger.info("Canonicizing all SMILES.")
with Parallel(n_jobs=12) as parallel:
    results = parallel(delayed(lambda x, y: (canonicize_smi(x), y))(smi, label) for smi, label in tqdm(zip(smi_list, labels)))
#combined = Parallel(n_jobs=12)((delayed(canonicize_smi)(smi), label) for smi, label in tqdm(zip(smi_list, labels)))

final_df = pd.DataFrame(results, columns=["Raw", "Labels"])
# this drops duplicate entries by canonical SMILES
final_df.drop_duplicates(["Raw"], inplace=True)
# this drops all the unwanted elements
final_df = final_df.loc[~final_df["Raw"].str.contains(r'\b(?:{})\b'.format('|'.join(exclude_list)))]
# reset numbers
final_df.reset_index(drop=True, inplace=True)

# calculating molecular weights
molecular_weights = list()
for smi in final_df["Raw"]:
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    molecular_weights.append(Descriptors.ExactMolWt(mol))
final_df["MW"] = molecular_weights

logger.info(f"Final size of dataset without duplication: {len(final_df)}.")

final_df.to_pickle("../data/processed/combined_smiles.pkl.bz2")
# save a separate SMILES file for mol2vec use
with open("../data/interim/collected_smiles.smi", "w+") as write_file:
    for smi in final_df["Raw"]:
        if smi != "":
            write_file.write(f"{smi}\n")
