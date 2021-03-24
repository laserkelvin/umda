
import h5py
from sklearn.decomposition import IncrementalPCA
from joblib import dump

data = h5py.File("../data/processed/smiles_embeddings_300.h5", "r")

# load mol2vec embeddings in
X = data["vectors"][:]
model = IncrementalPCA()

model.fit(X)
dump(model, "../models/pca_analysis_model.pkl")

