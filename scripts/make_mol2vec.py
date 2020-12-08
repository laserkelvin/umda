
from mol2vec import features
from gensim.models import word2vec

CORPUSNAME = "mol2vec_corpus.dat"
RADIUS = 1
NJOBS = 6

# create a corpus from the SMILES
features.generate_corpus("../data/interim/collected_smiles.smi", CORPUSNAME, RADIUS, sentence_type="alt", n_jobs=NJOBS, sanitize=False)

model = features.train_word2vec_model(CORPUSNAME, "../models/mol2vec_model.pkl", vector_size=300, min_count=1, n_jobs=NJOBS)
