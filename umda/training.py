
from typing import Tuple
from math import floor

import numpy as np
from scipy.stats import lognorm, uniform
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV, GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import kernels
from sklearn.utils import resample

# these imports are specifically for writing the custom CV
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils.validation import _num_samples


def random_cv_search(
    data, estimator, hparams, seed, n_splits: int = 5, **kwargs
):
    """
    Perform a randomized CV search for hyperparameter
    optimization. The main difference between this
    function and the grid search is that for continuous
    hyperparameter variables, we use uniform sampling
    between the min and max values. For categorical (specificallly
    string lists) we just feed the lists in directly.
    
    Kwargs are passed into the random CV object.

    Parameters
    ----------
    data : [type]
        [description]
    estimator : [type]
        [description]
    hparams : [type]
        [description]
    seed : [type]
        [description]
    n_splits : int, optional
        [description], by default 5

    Returns
    -------
    [type]
        [description]
    """
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


def grid_cv_search(data: Tuple[np.ndarray], estimator, hparams, seed: int, n_splits: int = 5, cv=None, **kwargs):
    kwargs.setdefault("n_jobs", 16)
    kwargs.setdefault("scoring", "r2")
    kwargs.setdefault("refit", False)
    try:
        X, y, indices = data
    except:
        X, y = data
        indices = None
    if cv is None:
        splitter = ShuffleSplit(n_splits, random_state=seed)
    else:
        splitter = cv
    grid_search = GridSearchCV(
        estimator,
        {f"regressor__{key}": value for key, value in hparams.items()},
        cv=splitter,
        **kwargs
    )
    result = grid_search.fit(X, y, indices)
    return result


def get_best_model(estimator, data: Tuple[np.ndarray], cv, **kwargs):
    kwargs.setdefault("n_jobs", 8)
    kwargs.setdefault("scoring", "neg_mean_squared_error")
    X, y, indices = data
    scores = cross_val_score(estimator, X, y, groups=indices, **kwargs)
    best_set = scores.argmax()
    # now get retrieve the dataset and refit
    for index, split in enumerate(cv.split(X, y, indices)):
        if index == best_set:
            train, test = split
            break
    return estimator.fit(X[train], y[train]), (train, test)


def get_molecule_split_bootstrap(data: Tuple[np.ndarray], seed: int, n_samples: int = 500, replace: bool = True, noise_scale: float = 0.5, molecule_split: float = 0.2, test_size: float = 0.2):
    """
    This function specifically splits the training set into train
    and validation sets within molecule classes. The idea behind this
    is to prevent data leakage.

    Parameters
    ----------
    data : Tuple[np.ndarray]
        [description]
    seed : int
        [description]
    n_samples : int, optional
        [description], by default 500
    replace : bool, optional
        [description], by default True
    noise_scale : float, optional
        [description], by default 0.5
    molecule_split : float, optional
        [description], by default 0.2
    """
    true_X, true_y = data
    indices = np.arange(len(true_y))
    rng = np.random.default_rng(seed)
    # shuffle the molecules
    rng.shuffle(indices)
    split_num = int(len(indices) * molecule_split)
    test_indices = indices[:split_num]
    train_indices = indices[split_num:]
    test_indices.sort(); train_indices.sort()
    sets = list()
    indices = list()
    for index_set, train in zip([train_indices, test_indices], [True, False]):
        if train:
            num_samples = int(n_samples * (1 - test_size))
        else:
            num_samples = int(n_samples * test_size)
        resampled_indices = resample(index_set, n_samples=num_samples, replace=replace, random_state=seed)
        resampled_indices.sort()
        resampled_X, resampled_y = true_X[resampled_indices], true_y[resampled_indices]
        reshuffled_indices = np.arange(resampled_y.size)
        rng.shuffle(reshuffled_indices)
        resampled_y += rng.normal(0., noise_scale, size=resampled_y.size)
        sets.append(
            (resampled_X[reshuffled_indices], resampled_y[reshuffled_indices])
        )
        indices.append(resampled_indices[reshuffled_indices])
    return sets, np.concatenate(indices)


def get_bootstrap_samples(data: Tuple[np.ndarray], seed: int, n_samples: int = 500, replace: bool = True, noise_scale: float = 0.5):
    """
    Wrapper function to bootstrap column densities in a dataset.
    The idea here is to generate "new" data by sampling with
    replacement, and adding Gaussian noise to the log column
    densities. The scale of the noise is set by the parameter
    `noise_scale`.

    Parameters
    ----------
    data : Tuple[np.ndarray]
        2-tuple containing X (2D) and y (1D) NumPy arrays
    seed : int
        Seed used to set the random state
    n_samples : int, optional
        Target dataset size, by default 500
    replace : bool, optional
        Whether to do bootstrapping with replacement, by default True
    noise_scale : float, optional
        Gaussian scale for target noise, by default 0.5

    Returns
    -------
    2-tuple
        boot_X is a 2D NumPy array of features with
        shape [N, L] where N = `n_samples`. boot_y
        is a NumPy 1D array with bootstrapped, noisy
        regression targets with shape [N].
    """
    boot_X, boot_y = resample(*data, n_samples=n_samples, replace=replace, random_state=seed)
    rng = np.random.default_rng(seed)
    boot_y += rng.normal(0., noise_scale, size=boot_y.size)
    return boot_X, boot_y


def bootstrap_fit(data, estimator, seed, n_samples: int = 500, test_size: float = 0.2, replace: bool = True, noise_scale: float = 0.5):
    X, y = data
    boot_X, boot_y = get_bootstrap_samples(data, seed, n_samples, replace, noise_scale)
    train_X, test_X, train_y, test_y = train_test_split(boot_X, boot_y, test_size=test_size, shuffle=True, random_state=seed)
    # fit to the bootstrapped training samples
    result = estimator.fit(train_X, train_y)
    # run standard metrics for performance
    train_error = mean_squared_error(train_y, estimator.predict(train_X))
    test_error = mean_squared_error(test_y, estimator.predict(test_X))
    r2 = r2_score(y, estimator.predict(X))
    return result, (train_error, test_error, r2), ((train_X, train_y), (test_X, test_y))


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
    """
    Generates a regression model using the sklearn pipeline
    pattern. This allows a preprocessing scaler normalization
    step prior to the estimator, allowing some easy flexibility
    for model testing.

    Parameters
    ----------
    base_estimator : [type]
        An instance of an sklearn estimator
    scale : bool, optional
        Whether to use `StandardScaler` to normalize
        the data prior to regression, by default False

    Returns
    -------
    sklearn `Pipeline` object
    """
    if scale:
        models = [("scaler", StandardScaler()), ("regressor", base_estimator)]
    else:
        models = [("regressor", base_estimator)]
    return Pipeline(models)


def get_gp_kernel():
    # gp_kernel = kernels.RBF() + kernels.ConstantKernel() * kernels.RationalQuadratic(alpha_bounds=(1e-3, 1e2))
    # gp_kernel = kernels.RationalQuadratic(alpha=100) * 1. + kernels.RBF() * 1e-2 + kernels.WhiteKernel()
    gp_kernel = kernels.RationalQuadratic() + kernels.DotProduct() + kernels.WhiteKernel()
    return gp_kernel


class BootstrappedCV(BaseCrossValidator):
    def __init__(self, indices, n_splits: int):
        self.indices = indices
        self.unique = np.unique(indices)
        self.splitter = KFold

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        train_group, test_group = indices[self.train_mask], indices[np.logical_not(self.train_mask)]
        min_train, max_train = floor(train_group.size * 0.25), floor(train_group.size * 0.75)
        min_test, max_test = floor(test_group.size * 0.25), floor(test_group.size * 0.75)
        for _ in range(self.n_splits):
            # generate a random number of samples to bootstrap
            # from either group uniformly
            num_train = self.random_state.randint(min_train, max_train)
            num_test = self.random_state.randint(min_test, max_test)
            train = self._bootstrap(train_group, num_train)
            test = self._bootstrap(test_group, num_test)
            yield train, test

    def _bootstrap(self, group: np.ndarray, n_samples: int):
        return resample(group, replace=False, n_samples=n_samples, random_state=self.random_state)

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        return self._iter_test_indices(X, y, groups)
            

class SamplesBootstrappedCV(BootstrappedCV):
    def __init__(self, train_mask: int, n_splits: int, seed: int, fraction: float):
        assert 0. <= fraction <= 1.
        super().__init__(train_mask, n_splits, seed)
        self.fraction = fraction

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        train_group, test_group = indices[self.train_mask], indices[np.logical_not(self.train_mask)]
        # number of training examples correspond to a fraction
        num_train = int(self.fraction * train_group.size)
        num_test = int(self.fraction * test_group.size)
        for _ in range(self.n_splits):
            # generate a specified number of examples
            train = self._bootstrap(train_group, num_train)
            test = self._bootstrap(test_group, num_test)
            yield train, test


def custom_learning_curve(estimator, data, true, fractions, cv):
    X, y, groups = data
    # the actual TMC-1 data
    true_X, true_y = true
    full_data = list()
    steps = (fractions * y.size).astype(int)
    for step in steps:
        step_data = list()
        temp_X, temp_y, temp_group = X[:step], y[:step], groups[:step]
        for split in cv.split(temp_X, temp_y, temp_group):
            train_index, test_index = split
            result = estimator.fit(temp_X[train_index], temp_y[train_index])
            step_data.append(mean_squared_error(true_y, result.predict(true_X)))
        full_data.append(np.array(step_data))
    return steps, np.vstack(full_data)
    