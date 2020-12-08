from typing import Dict, List
from collections import Counter

import h5py
import pandas as pd
import numpy as np
from joblib import dump, load
from loguru import logger
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import distance_metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from dask import array as da
from dask_ml import metrics

from umda import smi_vec


class EmbeddingModel(object):
    def __init__(self, w2vec_obj, transform=None, radius: int = 1) -> None:
        self._model = w2vec_obj
        self._transform = transform
        self._radius = radius
        self._covariance = None

    @property
    def model(self):
        return self._model

    @property
    def transform(self):
        return self._transform

    @property
    def radius(self):
        return self._radius

    def vectorize(self, smi: str):
        vector = smi_vec.smi_to_vector(smi, self.model, self.radius)
        # get the PCA embedding
        if self._transform is not None:
            model = self.transform.named_steps.get("incrementalpca", "pca")
            new_vector = model.transform(vector)
        else:
            new_vector = vector
        return new_vector

    def __call__(self, smi: str):
        return self.vectorize(smi)

    @classmethod
    def from_pkl(cls, w2vec_path, transform_path=None, **kwargs):
        w2vec_obj = smi_vec.load_model(w2vec_path)
        if transform_path:
            transform_obj = load(transform_path)
        else:
            transform_obj = None
        return cls(w2vec_obj, transform_obj, **kwargs)

    def save(self, path: str):
        dump(self, path)
        logger.info(f"Saved model to {path}.")


class MRS(object):
    def __init__(
        self, cluster_model, estimator, metric: str = "cosine", h5_file: str = None
    ) -> None:
        if h5_file:
            self._data = h5py.File(h5_file, "r")
        else:
            self._data = None
        self._cluster = cluster_model
        self._estimator = estimator
        self._distance = distance_metrics().get(metric)

    @classmethod
    def from_pickle(cls, cluster_path: str, estimator_path: str, **kwargs):
        cluster_model = load(cluster_path)
        estimator_model = load(estimator_path)
        return cls(cluster_model, estimator_model, **kwargs)

    def measure_distance(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return self._distance(A, B)

    def _find_top_recs(self, A: np.ndarray, B: np.ndarray, top: int = 10) -> np.ndarray:
        indices = list()
        distances = self.measure_distance(A, B)
        # loop over A sample dimensions, assuming A is the inventory
        # and get the `top` number of smallest distance
        for A_mol in distances:
            indices.append(np.argsort(A_mol)[:top])
        return np.hstack(indices)

    def cluster(self, X: np.ndarray) -> np.ndarray:
        return self._cluster.predict(X=X)

    def estimator(self, X: np.ndarray) -> Dict:
        if self.is_gp_estimator:
            result, cov_matrix = self._estimator.predict(X, return_cov=True)
            std = np.sqrt(np.diag(cov_matrix))
            self._covariance = cov_matrix
            return {"Abundance": result, "Uncertainty": std}
        else:
            result = self._estimator.predict(X)
            return {"Abundance": result}

    @property
    def is_gp_estimator(self) -> bool:
        return type(self._estimator) == GaussianProcessRegressor

    def recommend(
        self,
        h5_data=None,
        smi=None,
        X=None,
        embedding_model=None,
        top: int = 10,
        dist_thres: float = 1e-5,
        cluster: bool = True
    ):
        """
        Recommend molecules and predict their abundances, given a target inventory (i.e. what
        is present in the source/experiemnt) and a full `data`set to lookup.
        
        The inventory is specified with `Y` and either `smi` or `X`; the former is a 1D array
        containing abundances, and the latter two is either a list of SMILES strings or a
        2D array containing `mol2vec` embeddings. If `smi` is provided, the embeddings are
        generated using an `embedding_model`.

        Parameters
        ----------
        smi : [type], optional
            [description], by default None
        X : [type], optional
            [description], by default None
        embedding_model : [type], optional
            [description], by default None
        top : int, optional
            [description], by default 10
        data : np.ndarray, optional
            [description], by default None

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        Exception
            [description]
        Exception
            [description]
        """
        if h5_data is None:
            # load into memory because otherwise stuff will be slow for lookup!
            data_ref = self._data
        else:
            data_ref = h5_data
        dataset = data_ref["pca"][:] if "pca" in data_ref.keys() else data_ref["vectors"][:]
        all_smiles = data_ref["smiles"][:]
        nsamples, embedding_dim = dataset.shape
        if not smi and X is None:
            raise Exception(
                "No inputs provided; please provide something for me to work with!"
            )
        if smi and embedding_model:
            X = np.vstack([embedding_model(s) for s in smi])
        elif smi and not embedding_model:
            raise Exception("SMILES provided but no embedding model.")
        # determine where our molecules lie
        if not cluster:
            mask = np.ones(nsamples, dtype=bool)
        else:
            # match the input species into clusters
            cluster_ids = self.cluster(X)
            # match molecules
            mask = (data_ref["cluster_ids"][:] == np.unique(cluster_ids)[:,None]).any(axis=1)
            logger.info(f"Using the following clusters: {np.unique(cluster_ids)}")
        distances = self.measure_distance(X, dataset)
        # exclude molecules that are in both the dataset and inventory and merge
        # with the clustering mask
        exclude = ~(np.abs(distances) <= dist_thres).sum(axis=0).astype(bool)
        np.logical_and(mask, exclude, out=mask)
        # targets is the subset of molecules within the dataset that are
        # in the selected clusters
        targets = dataset[mask]
        logger.info(f"There are {len(targets)} molecules in this aggregate.")
        # Get indices of molecules that match, as well as a mask that excludes
        # overlap between inventory and dataset
        indices = self._find_top_recs(X, targets, top)
        unique_indices = np.unique(indices)
        unique_indices.sort()
        # get the embeddings and the SMILES associated with the recommendations
        recommend_X = dataset[unique_indices]
        recommend_smiles = all_smiles[unique_indices]
        # if estimator is a Gaussian process, get the uncertainties too
        estimator_result = self.estimator(recommend_X)
        # this gives the number of times each molecule has been recommended;
        # molecules that come up frequently are probably worth looking for
        instances = Counter(indices)
        results = pd.DataFrame.from_dict(estimator_result)
        results["Index"] = unique_indices
        results["SMILES"] = recommend_smiles
        # sort by the number of times each SMILES has appeared
        results["Counts"] = results["Index"].map(instances)
        results.sort_values(["Abundance"], ascending=False, inplace=True)
        results.reset_index(inplace=True, drop=True)
        return results

    def predict(self, X: np.ndarray = None, smi: List[str] = None, embedding_model=None) -> np.ndarray:
        if X is None and smi is None:
            raise Exception("No molecules specified, both X and smi are `None`!")
        if X is None:
            X = np.vstack([embedding_model(s) for s in smi])
        results = self.estimator(X)
        results["SMILES"] = smi
        return pd.DataFrame.from_dict(results)        

    def save(self, path: str):
        dump(self, path)
        logger.info(f"Saved MRS to {path}")


class DaskMRS(MRS):
    """
    Slightly modified version of the stock MRS, using Dask as a drop in replacement for
    the pre-screening and filtering. That way the full dataset doesn't have to be loaded
    into memory. The estimator portion remains NumPy based because sklearn isn't fully
    supported by Dask arrays.

    Parameters
    ----------
    MRS : [type]
        [description]
    """
    def __init__(self, cluster_model, estimator, metric: str = "euclidean", h5_file: str = None) -> None:
        super().__init__(cluster_model, estimator, metric=metric, h5_file=h5_file)
        self._distance = metrics.pairwise.euclidean_distances
        logger.warning("Dask version of MRS is being run; logging messages may not be accurate!")

    def _find_top_recs(self, A: da.array, B: da.array, top: int = 10) -> da.array:
        indices = list()
        distances = self.measure_distance(A, B)
        # loop over A sample dimensions, assuming A is the inventory
        # and get the `top` number of smallest distance
        for A_mol in distances:
            A_mol.compute_chunk_sizes()
            indices.append(A_mol.map_blocks(np.argsort)[:top])
        return da.hstack(indices)

    def recommend(
        self,
        h5_data=None,
        smi=None,
        X=None,
        embedding_model=None,
        top: int = 10,
        dist_thres: float = 1e-5,
        cluster: bool = True
    ):
        """
        Recommend molecules and predict their abundances, given a target inventory (i.e. what
        is present in the source/experiemnt) and a full `data`set to lookup.
        
        The inventory is specified with `Y` and either `smi` or `X`; the former is a 1D array
        containing abundances, and the latter two is either a list of SMILES strings or a
        2D array containing `mol2vec` embeddings. If `smi` is provided, the embeddings are
        generated using an `embedding_model`.

        Parameters
        ----------
        smi : [type], optional
            [description], by default None
        X : [type], optional
            [description], by default None
        embedding_model : [type], optional
            [description], by default None
        top : int, optional
            [description], by default 10
        data : np.ndarray, optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        if h5_data is None:
            # load into memory because otherwise stuff will be slow for lookup!
            data_ref = self._data
        else:
            data_ref = h5_data
        dataset = da.from_array(data_ref["pca"]) if "pca" in data_ref.keys() else da.from_array(data_ref["vectors"])
        all_smiles = data_ref["smiles"][:]
        nsamples, embedding_dim = dataset.shape
        if not smi and X is None:
            raise Exception(
                "No inputs provided; please provide something for me to work with!"
            )
        if smi and embedding_model:
            X = da.from_array(np.vstack([embedding_model(s) for s in smi]))
        elif smi and not embedding_model:
            raise Exception("SMILES provided but no embedding model.")
        # determine where our molecules lie
        if not cluster:
            logger.info(f"Not using clusters.")
            mask = da.ones(nsamples, dtype=bool)
        else:
            cluster_ids = self.cluster(X)
            ref_cluster_ids = da.from_array(data_ref["cluster_ids"])
            # negate the mask if we're clustering
            unique_ids = da.unique(cluster_ids)
            # unique_ids.compute_chunk_sizes()
            mask = (ref_cluster_ids == unique_ids[:,None]).any(axis=0)
            logger.info(f"Using the following clusters: {unique_ids}")
        distances = self.measure_distance(X, dataset)
        # exclude molecules that are in both the dataset and inventory and merge
        # with the clustering mask
        exclude = ~(da.fabs(distances) <= dist_thres).sum(axis=0).astype(bool)
        da.logical_and(mask, exclude, out=mask)
        # targets is the subset of molecules within the dataset that are
        # in the selected clusters
        targets = dataset[mask]
        logger.info(f"There are {targets.shape[0]} molecules in this aggregate.")
        # Get indices of molecules that match, as well as a mask that excludes
        # overlap between inventory and dataset
        indices = self._find_top_recs(X, targets, top).compute()
        # from this point onwards we revert back to NumPy for performance
        unique_indices = np.unique(indices)
        unique_indices.sort()
        # get the embeddings and the SMILES associated with the recommendations
        recommend_X = dataset[unique_indices]
        recommend_smiles = all_smiles[unique_indices]
        # if estimator is a Gaussian process, get the uncertainties too
        estimator_result = self.estimator(recommend_X)
        # this gives the number of times each molecule has been recommended;
        # molecules that come up frequently are probably worth looking for
        instances = Counter(indices)
        results = pd.DataFrame.from_dict(estimator_result)
        results["Index"] = unique_indices
        results["SMILES"] = recommend_smiles
        # sort by the number of times each SMILES has appeared
        results["Counts"] = results["Index"].map(instances)
        results.sort_values(["Abundance"], ascending=False, inplace=True)
        results.reset_index(inplace=True, drop=True)
        return results