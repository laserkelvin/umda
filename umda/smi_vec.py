from typing import List

import h5py
import numpy as np
from gensim.models import word2vec
from rdkit import Chem
from mol2vec import features
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def smi_to_vector(smi: str, model, radius: int = 1) -> List[np.ndarray]:
    """
    Given a model, convert a SMILES string into the corresponding
    NumPy vector.
    """
    # Molecule from SMILES will break on "bad" SMILES; this tries
    # to get around sanitization (which takes a while) if it can
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    # generate a sentence from rdkit molecule
    sentence = features.mol2alt_sentence(mol, radius)
    # generate vector embedding from sentence and model
    vector = features.sentences2vec([sentence], model)
    return vector


def inchi_to_smiles(inchi: str) -> str:
    inchi = str(inchi)
    if len(inchi) != 0:
        mol = Chem.MolFromInchi(inchi, sanitize=False, removeHs=False)
        if mol:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            return smiles
    else:
        return ""


def load_model(filepath: str):
    return word2vec.Word2Vec.load(filepath)


def parallel_smi_vectorization(
    smiles: List[str],
    model,
    h5_file,
    workers: int = 4,
    vec_length: int = 300,
    radius: int = 1,
):
    """
    This uses threading to perform the embedding and save it to the HDF5 dataset.
    Unfortunately, not appreciably faster, probably because you have to pickle the
    model and we're I/O limited.
    """
    vectors = Parallel(n_jobs=workers, prefer="threads")(
        delayed(smi_to_vector)(smi, model, radius) for smi in tqdm(smiles)
    )
    h5_file["vectors"] = np.vstack(vectors)
    dt = h5py.string_dtype()
    smiles = h5_file.create_dataset("smiles", (len(smiles),), dtype=dt, data=smiles)
#    h5_file["smiles"] = smiles


def serial_smi_vectorization(all_smiles, model, h5_ref, vec_length=300, radius=1):
    """
    This performs the SMILES vectorization in serial.
    
    This is actually quite fast, as the loop over SMILES strings should be fairly
    quick without any I/O limitation, either through pickling or because we are
    streaming data to the HDF5 file.

    Parameters
    ----------
    all_smiles : [type]
        [description]
    model : [type]
        [description]
    h5_ref : [type]
        [description]
    vec_length : int, optional
        [description], by default 300
    radius : int, optional
        [description], by default 1
    """
    # smiles are stored as strings, and so need some special treatment
    dt = h5py.string_dtype()
    smiles = h5_ref.create_dataset("smiles", (len(all_smiles),), dtype=dt)
    # vectorize all the SMILES strings, and then store it in the HDF5 array
    vec = np.vstack([smi_to_vector(smi, model, radius) for smi in tqdm(all_smiles)])
    h5_ref["vectors"] = vec
    for index, smi in enumerate(all_smiles):
        smiles[index] = smi


def canonicize_smi(smi: str) -> str:
    """
    Function to convert any SMILES string into its canonical counterpart.
    This ensures that all comparisons made subsequently are made with the
    same SMILES representation, if it exists.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        return None
