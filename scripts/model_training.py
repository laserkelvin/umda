from typing import Tuple, Dict

import h5py
import pandas as pd
import numpy as np
from loguru import logger
from ruamel.yaml import YAML
from joblib import load, dump
from umda import EmbeddingModel
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

USE_DASK = False

models = {
    "linear_regression": [
        LinearRegression(fit_intercept=False),
        [{"normalize": [True, False],}],
    ],
    "svr": [
        SVR(),
        [
            {
                "kernel": ["rbf",],# "poly"],
                #"degree": [2, 3, 4],
                "C": 10**np.linspace(1.5, 2., 20),
                "gamma": ["auto", 0.05, 0.1],
                "epsilon": 10**np.linspace(-3., -1., 20),
            }
        ],
    ],
    "knn": [
        KNeighborsRegressor(),
        [
            {
                "n_neighbors": [2, 4, 10, 15, 30, 50, 70],
                "metric": ["cosine", "euclidean",],
                "weights": ["uniform", "distance"]
            }
        ],
    ],
    "rfr": [
        RandomForestRegressor(max_features=None, random_state=1205),
        [
            {"n_estimators": [10, 20, 50, 80, 100, 125, 150, 200],
             "max_leaf_nodes": [None, 5, 10, 15, 20, 40],
             "min_samples_leaf": [1, 3, 5, 10, 15, 20, 25, 35],
             "max_depth": [None, 5, 10, 15, 20]
             }
        ],
    ],
    "gbr": [
        GradientBoostingRegressor(random_state=1205),
        [
            {
                "learning_rate": 10 ** np.linspace(-3.0, 1.0, 20),
                "n_estimators": [5, 10, 30, 50, 80, 100, 125, 150, 200],
                "subsample": [0.2, 0.4,  0.6, 0.8, 1.],
                "max_depth": [1, 2, 3, 4, 5, 6]
            }
        ],
    ],
    "gpr": [
        None,
        [{"alpha": 10 ** np.linspace(-10.0, 1.0, 5), "n_restarts_optimizer": [5, 10, 15, 20]}],
    ],
}


def standardize_test(
    estimator: "sklearn model",
    search_params: Tuple[Dict],
    data: Tuple[np.ndarray, np.ndarray],
    seed: int = 42,
    n_jobs: int = 8,
    cv: int = 20
):
    # split data into X and y for regression
    X, y = data
    # Manually specify 10-fold cross-validation for the grid search
    kfold = KFold(cv, random_state=seed, shuffle=True)
    grid_search = GridSearchCV(
        estimator,
        search_params,
        scoring="neg_mean_squared_error",
        cv=kfold,
        n_jobs=n_jobs,
    )
    # run the grid search
    grid_search.fit(X, y)
    # give some summary statistics
    y_mask = y != 0.0
    y_pred = grid_search.best_estimator_.predict(X)
    # masked error is excluding negative examples
    mse = metrics.mean_squared_error(y_pred, y)
    masked_mse = metrics.mean_squared_error(y_pred[y_mask], y[y_mask])
    r2 = metrics.r2_score(y, y_pred)
    errors = {"mse": float(mse), "masked_mse": float(masked_mse), "r^2": float(r2)}
    return grid_search, grid_search.best_estimator_, errors


def mask_distant_species(
        target: np.ndarray, fullset: np.ndarray, upper_percentile: float = 97.
) -> np.ndarray:
    distances = cosine_distances(target, fullset)
    logger.info(f"Min/max distance: {distances.min()}/{distances.max()}")
    logger.info(f"Mean/std distance: {distances.mean()}/{distances.std()}")
    lower, mean, upper = np.percentile(distances, [3., 50., upper_percentile])
    logger.info(f"3%/50%/{upper_percentile}%: {lower:.3f}/{mean:.3f}/{upper:.3f}")
    dist_mask = distances.mean(axis=0) > upper
    return dist_mask


def main(
    prediction_output: str,
    seed: int = 42,
    distance_threshold: float = 0.8,
    n_jobs: int = 8,
    cv: int = 10
):
    logger.add("model_training.log")
    logger.info(f"Using seed {seed}, cosine distance zeroing: {distance_threshold}")
    logger.info(f"Cross-validation will be done with {n_jobs} workers.")
    #rng = np.random.default_rng(seed)
    logger.info("Loading data")
    # prepare and load data
    data = h5py.File("../data/processed/pipeline_embeddings_70.h5", "r")
    original = h5py.File("../data/processed/smiles_embeddings_300.h5", "r")
    pipeline = load("../models/embedding_pipeline.pkl")
    pca = load("../models/pca_model.pkl")
    embedding_model = load("../models/EmbeddingModel.pkl")
    ## load in the TMC-1 data and grab the embedding vectors
    tmc1_df = pd.read_pickle("../data/processed/tmc1_ready.pkl")
    # ignore H2 lol
    #tmc1_df = tmc1_df.loc[tmc1_df["canonical"] != "[HH]"]
    tmc1_df.reset_index(inplace=True, drop=True)
    ## get into NumPy array
    #tmc1_vecs = np.vstack(tmc1_df["vectors"])
    ##indices = np.arange(len(data["pca"]))
    #for step in pipeline.steps[:2]:
    #    tmc1_vecs = step[1].transform(tmc1_vecs)
    ## get the TMC-1 cluster IDs
    #tmc1_cluster_ids = pipeline.steps[-1][1].predict(tmc1_vecs)
    tmc1_vecs = np.vstack([embedding_model.vectorize(smi) for smi in tmc1_df["canonical"]])
    tmc1_cluster_ids = np.array([embedding_model.cluster(smi) for smi in tmc1_df["canonical"]])
    #if USE_DASK:
    #    tmc1_cluster_ids = tmc1_cluster_ids.compute()
    ## holdout_cluster_ids = pipeline.predict(holdout_vecs).compute()
    ## compute the PCA embedding for the TMC-1 molecules
    #tmc1_embedding = pipeline.steps[0][1].transform(tmc1_vecs)
    # holdout_embedding = pipeline.steps[0][1].transform(holdout_vecs)
    # for computational efficiency, just grab the most relevant
    # molecules to TMC-1
    mask = np.zeros_like(data["cluster_ids"], dtype=bool)
    for i in np.unique(tmc1_cluster_ids):
        mask += data["cluster_ids"][:] == i
    logger.info(f"There are {mask.sum()} molecules in the TMC-1 cluster(s)")
    # Extract out the molecules contained within our cluster
    all_pca = (data["pca"][:])[mask, :]
    logger.info(f"Shape of the PCA vectors: {all_pca.shape}")
    logger.info(f"Shape of the TMC1-1 vectors: {tmc1_vecs.shape}")
    pca_dim = all_pca.shape[-1]
#    subset_smiles = (data["smiles"][:])[mask]
    # set them as "X" and "Y" for ease of reference
    X = tmc1_vecs.copy()
    Y = np.log10(tmc1_df["Column density (cm^-2)"].to_numpy())
    # convert to abundance
    #Y = tmc1_df["Column density (cm^-2)"].to_numpy() / 1e22
    # what we want to do now is to set molecules we have little chance of
    # detecting to have zero column densities
    dist_mask = mask_distant_species(X, all_pca, distance_threshold)
    dummies = all_pca[dist_mask,:]
    logger.info(f"Setting {dist_mask.sum()} entries to zero column density.")
#    logger.info(f"Examples of excluded molecules: {subset_smiles[dist_mask][:5]}")
    dummy_y = np.zeros(dummies.shape[0])
    logger.info("Preparing training data")
    # add the constrained values to our training data
    train_x = np.vstack([X, dummies])
    train_y = np.hstack([Y, dummy_y])
    logger.info(f"Shape of X: {train_x.shape} and Y: {train_y.shape}")
    results = dict()
    with h5py.File(prediction_output, "a") as h5_output:
        try:
            del h5_output["tmc1_cluster_mask"]
        except:
            pass
        # save the intercluster mask
        h5_output["tmc1_cluster_mask"] = mask
        # now do the standardized training and testing for every model
        for model_name, conditions in models.items():
            # see if we can delete the key
            try:
                del h5_output[model_name]
            except KeyError:
                pass
            logger.info(f"Performing {cv}-fold CV on {model_name}")
            model, hyperparams = conditions
            # for gaussian process, define the covariance function
            if model_name == "gpr":
                kernel = kernels.ConstantKernel() * kernels.RBF(
                    3.0, (1e-1, 10.0)
                ) + kernels.RationalQuadratic(
                    200.0, 20.0, alpha_bounds=(1e-3, 5e2), length_scale_bounds=(50.0, 1e4)
                    ) * kernels.ConstantKernel() + kernels.ConstantKernel()
                model = GaussianProcessRegressor(kernel, random_state=1205)
            grid, best_model, errors = standardize_test(
                model, hyperparams, (train_x, train_y), n_jobs=n_jobs, cv=cv, seed=seed
            )
            # log the model results
            results[model_name] = errors
            logger.info(f"Best errors for {model_name}: {errors}")
            # pickle the CV grid
            dump(grid, f"../models/{model_name}_grid.pkl")
            cv_df = pd.DataFrame.from_dict(grid.cv_results_)
            cv_df.to_csv(f"../models/{model_name}_grid_summary.csv", index=False)
            logger.info(f"Caching predictions for best model")
            if model_name != "gpr":
                pred_Y = best_model.predict(all_pca)
                h5_output[f"{model_name}"] = pred_Y
            else:
                pred_Y, pred_std = best_model.predict(all_pca, return_std=True)
                gpr_tmc_y, gpr_tmc_cov = best_model.predict(
                    X, return_cov=True
                )
                # save a bunch of stuff for Gaussian Process
                for target, name in zip(
                    [pred_Y, pred_std, gpr_tmc_y, gpr_tmc_cov],
                    ["all", "all_std", "tmc_reproduction", "tmc_cov"],
                ):
                    try:
                        del h5_output[f"{model_name}_{name}"]
                    except KeyError:
                        pass
                    h5_output[f"{model_name}_{name}"] = target
            tmc1_df[model_name] = best_model.predict(X)
            tmc1_df.to_csv("tmc1_results.csv", index=False)
    # save the errors for later reporting
    yaml = YAML()
    with open("../models/training_errors.yml", "w+") as write_file:
        yaml.dump(results, write_file)


if __name__ == "__main__":
    params = {
        "prediction_output": "../data/processed/model_predictions.h5",
        "seed": 42,
        "distance_threshold": 99.98,
        "n_jobs": 16,
        "cv": 10
    }
    main(**params)
