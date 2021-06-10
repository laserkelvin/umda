import numpy as np
from scipy.stats import lognorm, uniform
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import kernels
from sklearn.utils import resample


def random_cv_search(
    data, estimator, hparams, seed, n_splits: int = 5, **kwargs
):
    kwargs.setdefault("n_jobs", 16)
    kwargs.setdefault("scoring", "r2")
    kwargs.setdefault("n_iter", 500)
    kwargs.setdefault("refit", False)
    X, y = data
    splitter = ShuffleSplit(n_splits, random_state=seed)
    # determine the distributions over which to do the CV search
    distributions = dict()
    for key, value in hparams.items():
        if all([type(val) == str for val in value]):
            distributions[f"regressor__{key}"] = value
        elif all([type(val) == bool for val in value]):
            distributions[f"regressor__{key}"] = value
        else:
            # remove all the string types
            temp = list()
            for val in value:
                try:
                    temp.append(float(val))
                except (ValueError, TypeError):
                    pass
            # draw Gaussian distribution
            distributions[f"regressor__{key}"] = uniform(np.min(temp), np.max(temp))
    # do a randomized CV search using R^2 as the objective
    grid_search = RandomizedSearchCV(
        estimator,
        distributions,
        cv=splitter,
        **kwargs
    )
    result = grid_search.fit(X, y)
    return result


def grid_cv_search(data, estimator, hparams, seed, n_splits: int = 5, **kwargs):
    kwargs.setdefault("n_jobs", 16)
    kwargs.setdefault("scoring", "r2")
    kwargs.setdefault("refit", False)
    X, y = data
    splitter = ShuffleSplit(n_splits, random_state=seed)
    grid_search = GridSearchCV(
        estimator,
        {f"regressor__{key}": value for key, value in hparams.items()},
        cv=splitter,
        **kwargs
    )
    result = grid_search.fit(X, y)
    return result


def bootstrap_fit(data, estimator, seed, n_samples: int = 500, test_size: float = 0.2, replace: bool = True):
    X, y = data
    boot_X, boot_y = resample(X, y, n_samples=n_samples, replace=replace, random_state=seed)
    train_X, test_X, train_y, test_y = train_test_split(boot_X, boot_y, test_size=test_size, shuffle=True, random_state=seed)
    # fit to the bootstrapped training samples
    result = estimator.fit(train_X, train_y)
    # run standard metrics for performance
    train_error = mean_squared_error(train_y, estimator.predict(train_X))
    test_error = mean_squared_error(test_y, estimator.predict(test_X))
    r2 = r2_score(y, estimator.predict(X))
    return result, (train_error, test_error, r2)


def standardized_fit_test(
    data, estimator, hparams, seed, n_splits: int = 100, test_size: float = 0.2
):
    X, y = data
    splitter = ShuffleSplit(n_splits, test_size=test_size, random_state=seed)
    best_train_error, best_test_error, best_performance = np.inf, np.inf, np.inf
    log = list()
    for split_index, (train_index, test_index) in enumerate(splitter.split(X, y)):
        train_X, test_X, train_y, test_y = (
            X[train_index],
            X[test_index],
            y[train_index],
            y[test_index],
        )
        # current_estimator = estimator.__class__()
        # set the estimator hyperparameters
        estimator.__dict__.update(
            **{f"regressor__{key}": value for key, value in hparams.items()}
        )
        result = estimator.fit(train_X, train_y)
        # compute the mean squared error
        train_error = mean_squared_error(train_y, result.predict(train_X))
        test_error = mean_squared_error(test_y, result.predict(test_X))
        performance = np.abs(train_error - test_error) / (train_error + test_error)
        r2 = r2_score(y, result.predict(X))
        log.append(
            {
                "train_error": train_error,
                "test_error": test_error,
                "performance": performance,
                "r2": r2,
                "train_index": train_index,
                "test_index": test_index,
            }
        )
        if test_error < best_test_error:
            best_split = (train_index, test_index)
            best_train_error = train_error
            best_test_error = test_error
            best_performance = performance
            best_estimator = result
    return best_estimator, best_train_error, best_test_error, best_performance, best_split, log


def compose_model(base_estimator, scale: bool = False):
    if scale:
        models = [("scaler", StandardScaler()), ("regressor", base_estimator)]
    else:
        models = [("regressor", base_estimator)]
    return Pipeline(models)


def get_gp_kernel():
    gp_kernel = kernels.ConstantKernel() * \
    kernels.RBF(10.0, (1e-1, 30.0)) + \
    kernels.RationalQuadratic(200.0, 20.0, alpha_bounds=(1e-3, 5e2), length_scale_bounds=(50.0, 1e4)) * \
        kernels.ConstantKernel() + kernels.ConstantKernel()
    return gp_kernel