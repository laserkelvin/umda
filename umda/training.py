import numpy as np
from scipy.stats import lognorm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def random_cv_search(
    data, estimator, hparams, seed, n_splits: int = 5, n_jobs: int = 16
):
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
            distributions[f"regressor__{key}"] = lognorm(np.mean(temp), np.std(temp))
    # do a randomized CV search using R^2 as the objective
    grid_search = RandomizedSearchCV(
        estimator,
        distributions,
        n_iter=1000,
        scoring="r2",
        cv=splitter,
        n_jobs=n_jobs,
        refit=False,
    )
    result = grid_search.fit(X, y)
    return result


def grid_cv_search(data, estimator, hparams, seed, n_splits: int = 5, n_jobs: int = 16):
    X, y = data
    splitter = ShuffleSplit(n_splits, random_state=seed)
    grid_search = GridSearchCV(
        estimator,
        {f"regressor__{key}": value for key, value in hparams.items()},
        scoring="r2",
        cv=splitter,
        n_jobs=n_jobs,
        refit=False,
    )
    result = grid_search.fit(X, y)
    return result


def standardized_fit_test(
    data, estimator, hparams, seed, n_splits: int = 100, test_size: float = 0.2
):
    X, y = data
    splitter = ShuffleSplit(n_splits, test_size=test_size, random_state=seed)
    best_train_error, best_test_error, best_combined = np.inf, np.inf, np.inf
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
        combined_error = mean_squared_error(y, result.predict(X))
        r2 = r2_score(y, result.predict(X))
        log.append(
            {
                "train_error": train_error,
                "test_error": test_error,
                "combined_error": combined_error,
                "r2": r2,
                "train_index": train_index,
                "test_index": test_index,
            }
        )
        if test_error < best_test_error:
            best_split = (train_index, test_index)
            best_train_error = train_error
            best_test_error = test_error
            best_combined = combined_error
            best_estimator = result
    return best_estimator, best_train_error, best_test_error, best_split, log


def compose_model(base_estimator, scale: bool = False):
    if scale:
        models = [("scaler", StandardScaler()), ("regressor", base_estimator)]
    else:
        models = [("regressor", base_estimator)]
    return Pipeline(models)
