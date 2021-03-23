import numpy as np
import pandas as pd
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from rdkit import Chem
from umda import EmbeddingModel
from umda import compute
import re
from joblib import parallel_backend, load


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


kida_mols = pd.read_csv("../../data/external/kida-molecules_05_Jul_2020.csv").dropna()


kida_mols = kida_mols[kida_mols["InChI"] != "InChI="]


kida_mols.reset_index(drop=True, inplace=True)


# In[6]:


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


# In[7]:


kida_mols["SMILES"] = kida_mols["InChI"].map(mapping)


# ### Load in the KIDA reactions

# In[8]:


reactions = parse_kida_format("../../data/external/kida.uva.2014/kida.uva.2014.dat")
reactions_df = kida_to_dataframe(reactions)


# In[9]:


# vectorize the function
compute_rate = np.vectorize(compute.compute_rate)


# In[10]:


# convert the standard formulae into SMILES
reactions_df["A_smi"] = reactions_df["A"].map(mapping)
reactions_df["B_smi"] = reactions_df["B"].map(mapping)


# In[11]:


valid_reactions = reactions_df.dropna()
valid_reactions.reset_index(drop=True, inplace=True)


# In[12]:


# load in the EmbeddingModel class predumped
embedder = load("../../models/EmbeddingModel.pkl")


# ### Generate all the kinetic data, and put it into a combined table

# In[19]:


X, y = list(), list()
for index, row in valid_reactions.iterrows():
    min_temp, max_temp = row["min_temp"], row["max_temp"]
    # force the temperature ranges to be within [0, 300]
    min_temp = max(0.0, min_temp)
    max_temp = min(300.0, max_temp)
    temperatures = np.linspace(min_temp, max_temp, 30)
    rates = compute_rate(
        row["react_class"], temperatures, row["alpha"], row["beta"], row["gamma"]
    )
    A, B = embedder.vectorize(row["A_smi"])[0], embedder.vectorize(row["B_smi"])[0]
    for rate, temp in zip(rates, temperatures):
        X.append(np.concatenate((A, B, [temp])))
        y.append(rate)


# In[21]:


# stack up all the embeddings into arrays
# A_emb = np.vstack([embedder.vectorize(smi)[0] for smi in valid_reactions["A_smi"].values])
# B_emb = np.vstack([embedder.vectorize(smi)[0] for smi in valid_reactions["B_smi"].values])


# In[76]:


# combined = np.hstack([A_emb, B_emb])


# In[22]:


# The nominal product given as the row-wise sum of A + B
# product = A_emb + B_emb


# In[47]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV


# In[80]:


model = GradientBoostingRegressor(learning_rate=0.01, max_depth=3)


# In[79]:


rates = valid_reactions["Rate"].values


# In[81]:


model.fit(combined, rates)


# In[82]:


mean_squared_error(model.predict(combined), rates)


# ## K-Fold cross-validation for model selection

# In[77]:


kfold = KFold(5, random_state=42, shuffle=True)


# In[62]:


hyperparams = {
    "learning_rate": 10 ** np.arange(-4.0, 0.0, 1.0),
    "n_estimators": [
        20,
        50,
        100,
        150,
    ],
    "max_depth": [1, 3, 5],
}


# In[63]:


search = GridSearchCV(
    GradientBoostingRegressor(), hyperparams, cv=kfold, scoring="neg_mean_squared_error"
)


# In[64]:


with parallel_backend("threading", n_jobs=4):
    result = search.fit(combined, rates)


# In[70]:


herp = result.cv_results_
del herp["params"]


# In[72]:


pd.DataFrame(herp).sort_values(["rank_test_score"])


# In[ ]:
