from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import h5py
import numpy as np
import pandas as pd
from joblib import dump, parallel_backend
from loguru import logger


data = h5py.File("../data/processed/kida_rate_data.h5", "r")

X = data["X"][:]
y = data["y"][:]
# floated the idea of doing stuff in logspace, but it doesn't seem to work
# as well as linear space
#y = np.log10(data["y"][:])
#y[np.isinf(y)] = 0.

scaler = StandardScaler()
trans_y = scaler.fit_transform(y[:,None]).ravel()

# do a K-fold split on the data, and check for the best validation score
#kfold = KFold(5, random_state=42, shuffle=True)
#hyperparams = {
#    "learning_rate": [0.1, 0.3, 0.5,],
#    "subsample": [0.5, 0.7, 1.0],
#    "max_depth": [1, 3, 5, 7, 10, 15],
#    "n_estimators": [10, 30, 50, 70, 100, 150, 200],
#}
#search = GridSearchCV(
#    GradientBoostingRegressor(), hyperparams, cv=kfold, scoring="neg_mean_squared_error"
#)
#
#with parallel_backend("threading", n_jobs=24):
#    result = search.fit(X, trans_y)

train_X, test_X, train_y, test_y = train_test_split(X, trans_y, test_size=0.3)
model = GradientBoostingRegressor(learning_rate=0.05, subsample=0.5, max_depth=7, n_estimators=250)

logger.info(f"Train/test sizes: {train_y.size}, {test_y.size}")
model.fit(train_X, train_y)
logger.info("Fit complete.")
train_error = mean_squared_error(scaler.inverse_transform(model.predict(train_X)), scaler.inverse_transform(train_y))
test_error = mean_squared_error(scaler.inverse_transform(model.predict(test_X)), scaler.inverse_transform(test_y))

logger.info(f"Train/test MSE: {train_error:.3e}, {test_error:.3e}.")

#dump(result, "../models/rate_gbr_history.pkl")
dump(model, "../models/rate_gbr_model.pkl")
pipeline = make_pipeline(scaler, model)
dump(pipeline, "../models/rate_gbr_pipeline.pkl")
logger.info("Finished")
#summary = result.cv_results_
#del summary["params"]
#pd.DataFrame(summary).to_csv("../models/rate_gbr_search.csv", index=False)
