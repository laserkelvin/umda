import numpy as np
import pandas as pd
from rdkit import Chem
from umda import EmbeddingModel
from umda import compute
import re
from joblib import parallel_backend, load
from tqdm.auto import tqdm
import h5py


def parse_kida_format(path: str):
    data = list()
    with open(path) as read_file:
        for line in read_file.readlines():
            if not line.startswith("!"):
                split_line = line.split()
                react1, react2 = split_line[:2]
                # this regex is complicated because sometimes people
                # provide inconsistent formatting on the exponent
                values = re.findall(r"\d.\d{3,4}[eE][\+\-]\d{1,2}", line)
                alpha, beta, gamma = [float(value) for value in values]
                # grab the second integer specification, which is the reaction type
                integers = re.findall(r"\s+\d{1,2}\s+", line)
                react_index = int(integers[1])
                min_temp, max_temp = float(split_line[-6]), float(split_line[-5])
                data.append(
                    [
                        react1,
                        react2,
                        alpha,
                        beta,
                        gamma,
                        react_index,
                        min_temp,
                        max_temp,
                    ]
                )
    return data


def kida_to_dataframe(
    data,
    ignore=["CRP", "CR", "Photon", "GRAIN0", "GRAIN-", "XH"],
    react_class=[3, 4, 5],
) -> pd.DataFrame:
    df = pd.DataFrame(
        data,
        columns=[
            "A",
            "B",
            "alpha",
            "beta",
            "gamma",
            "react_class",
            "min_temp",
            "max_temp",
        ],
    )
    # ignore reactions with reactants that are in the ignore list, and ensure that
    # we have a mapping for the reaction class
    filtered_df = df.loc[
        (~df["B"].str.contains(rf"\b(?:{'|'.join(ignore)})\b"))
        & (~df["A"].str.contains(rf"\b(?:{'|'.join(ignore)})\b"))
        & (df["react_class"].isin(react_class))
    ]
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


kida_mols = pd.read_csv("../data/external/kida-molecules_05_Jul_2020.csv").dropna()
kida_mols = kida_mols[kida_mols["InChI"] != "InChI="]
kida_mols.reset_index(drop=True, inplace=True)

# generate the mapping between "standard" formulae to SMILES
# by going through InChI provided by KIDA
mapping = dict()
with open("kida.inchi", "w+") as durr:
    for index, row in kida_mols.iterrows():
        inchi = row["InChI"]
        if "InChI" not in inchi:
            inchi = f"InchI={inchi.replace('Inchi', '')}"
        try:
            smi = Chem.MolToSmiles(
                Chem.MolFromInchi(inchi, sanitize=False), canonical=True
            )
            mapping[row["Formula"]] = smi
        except:
            print(f"{inchi} was untranscribable.")

# test run to make sure the mapping works
kida_mols["SMILES"] = kida_mols["InChI"].map(mapping)

reactions = parse_kida_format("../data/external/kida.uva.2014/kida.uva.2014.dat")
reactions_df = kida_to_dataframe(reactions)

# vectorize the rate function
compute_rate = np.vectorize(compute.compute_rate)

# convert the standard formulae into SMILES
reactions_df["A_smi"] = reactions_df["A"].map(mapping)
reactions_df["B_smi"] = reactions_df["B"].map(mapping)

# dropna removes all the molecules that do not have a valid mapping,
# as well as generally bad rows
valid_reactions = reactions_df.dropna()
valid_reactions.reset_index(drop=True, inplace=True)

# load in the EmbeddingModel class predumped
embedder = load("../models/EmbeddingModel.pkl")

# ### Generate all the kinetic data, and put it into a combined table
print("Generating rate coefficients")
X, y = list(), list()
for index, row in tqdm(valid_reactions.iterrows()):
    min_temp, max_temp = row["min_temp"], row["max_temp"]
    # force the temperature ranges to be within [0, 300]
    min_temp = max(0.0, min_temp)
    max_temp = min(300.0, max_temp)
    t_range = sorted([min_temp, max_temp])
    temperatures = np.linspace(*t_range, 50)
    try:
        rates = compute_rate(
            row["react_class"], temperatures, row["alpha"], row["beta"], row["gamma"]
        )
        A, B = embedder.vectorize(row["A_smi"]), embedder.vectorize(row["B_smi"])
        for rate, temp in zip(rates, temperatures):
            X.append(np.concatenate((A, B, [temp])))
            y.append(rate)
    except ValueError:
        pass

print("Finalizing arrays")
# finalize the arrays
X = np.vstack(X)
y = np.asarray(y)

print("Saving arrays to HDF5 file")
with h5py.File("../data/processed/kida_rate_data.h5", "a") as h5_file:
    for key, array in zip(["X", "y"], [X, y]):
        try:
            del h5_file[key]
        except KeyError:
            pass
        h5_file.create_dataset(key, data=array)
