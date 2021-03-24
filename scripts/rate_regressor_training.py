
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
import h5py
import numpy as np
import pandas as pd
from joblib import dump, parallel_backend


data = h5py.File("../data/processed/kida_rate_data.h5", "r")

X = data["X"][:]
y = data["y"][:]
#y = np.log10(data["y"][:])
#y[np.isinf(y)] = 0.

# do a K-fold split on the data, and check for the best validation score
kfold = KFold(5, random_state=42, shuffle=True)
hyperparams = {"learning_rate": [0.1, 0.3, 0.5, 1.], "subsample": [0.5, 0.7, 1.], "max_depth": [1, 3, 5, 7, 10, 15]}
search = GridSearchCV(GradientBoostingRegressor(), hyperparams, cv=kfold, scoring="neg_mean_squared_error")

with parallel_backend("threading", n_jobs=24):
    result = search.fit(X, y)

dump(result, "../models/rate_gbr_history.pkl")

summary = result.cv_results_
del summary["params"]
pd.DataFrame(summary).to_csv("../models/rate_gbr_search.csv", index=False)

