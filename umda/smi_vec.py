from typing import List

import numpy as np
from gensim.models import word2vec
from rdkit import Chem
from mol2vec import features


def smi_to_vector(smi: str, model, radius: int = 1) -> List[np.ndarray]:
    """
    Given a model, convert a SMILES string into the corresponding
    NumPy vector.
    """
    # Molecule from SMILES will break on "bad" SMILES; this tries
    # to get around sanitization (which takes a while) if it can
    try:
        mol = Chem.MolFromSmiles(smi)
    except RuntimeError:
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


def parallel_smi_vectorization(all_smiles, model, h5_ref, workers=4, vec_length=300):
    """
    This uses threading to perform the embedding and save it to the HDF5 dataset.
    Unfortunately, not appreciably faster :P
    """
    dataset = h5_ref.require_dataset(
        "vectors", (len(all_smiles), vec_length), dtype=np.float32, chunks=(10000, 300)
    )
    with ThreadPoolExecutor(workers) as pool:
        for index, result in enumerate(
            pool.map(smi_to_vector, all_smiles, (model for _ in range(len(all_smiles))))
        ):
            dataset[i, :] = result


def serial_smi_vectorization(all_smiles, model, h5_ref, vec_length=300):
    vectors = h5_ref.require_dataset(
        "vectors",
        (len(all_smiles), vec_length),
        dtype=np.float32,
        chunks=(10000, vec_length),
    )
    dt = h5py.string_dtype()
    smiles = h5_ref.create_dataset("smiles", (len(all_smiles),), dtype=dt)
    for index, result in enumerate(
        map(smi_to_vector, all_smiles, (model for _ in range(len(all_smiles))))
    ):
        vectors[index, :] = result
        smiles[index] = all_smiles[index]
