from mol2vec import features
from gensim.models import word2vec
from loguru import logger

logger.add("mol2vec_training.log")

CORPUSNAME = "mol2vec_corpus.dat"
RADIUS = 1
NJOBS = 32

logger.info(f"Using {NJOBS} workers, radius of {RADIUS}")
logger.info("Generating word2vec corpus")

with open("../data/interim/collected_smiles.smi", "r") as read_file:
    logger.info(f"Number of SMILES entries: {len(read_file.readlines())}")

# create a corpus from the SMILES
features.generate_corpus(
    "../data/interim/collected_smiles.smi",
    CORPUSNAME,
    RADIUS,
    sentence_type="alt",
    n_jobs=NJOBS,
    sanitize=False,
)

logger.info("Training mol2vec model")
model = features.train_word2vec_model(
    CORPUSNAME,
    "../models/mol2vec_model.pkl",
    vector_size=300,
    min_count=1,
    n_jobs=NJOBS,
)
