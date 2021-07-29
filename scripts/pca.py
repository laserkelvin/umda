
import h5py
from sklearn.decomposition import IncrementalPCA
from joblib import dump

from umda.utils import paths

"""
This trains a PCA model on the embeddings, and saves
the model for analysis; this is primarily to make that
figure showing the amount of explained variance as a
function of the number of components/dimensions kept.
"""

data = h5py.File(paths.get("processed").joinpath("smiles_embeddings_300.h5", "r")

# load mol2vec embeddings in
X = data["vectors"][:]
model = IncrementalPCA()

model.fit(X)
dump(model, "../models/pca_analysis_model.pkl")

