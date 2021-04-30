from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
import h5py
import numpy as np
import pandas as pd
from joblib import dump, parallel_backend
from loguru import logger

logger.add("rate_gp_training.log")

data = h5py.File("../data/processed/kida_rate_data.h5", "r")

X = data["X"][:]
logger.info(f"Shape of X: {X.shape}")
y = data["y"][:]
#y = np.log10(data["y"][:])
#y[np.isinf(y)] = 0.

scaler = StandardScaler()
trans_y = scaler.fit_transform(y[:,None]).ravel()

kernel = kernels.ConstantKernel() * kernels.RBF(3., (1e-1, 30.0)) + kernels.ConstantKernel() * kernels.RationalQuadratic(200., 20., alpha_bounds=(1e-3, 5e2), length_scale_bounds=(5., 1e5))
model = GaussianProcessRegressor(kernel, n_restarts_optimizer=5, random_state=42)

logger.info(f"Training {model} with kernel {kernel}.")
# do a K-fold split on the data, and check for the best validation score
#kfold = KFold(5, random_state=42, shuffle=True)
#hyperparams = {
#    "alpha": 10**np.linspace(-6., 0., 5), "n_restarts_optimizer": [5, 10, 15, 20, 40]
#}
#search = GridSearchCV(
#    GaussianProcessRegressor(kernel), hyperparams, cv=kfold
#)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.98)
logger.info(f"Train/test sizes: {len(train_X)}, {len(test_X)}")

model.fit(train_X, train_y)
logger.info(f"Fit done! Optimized kernel: {model.kernel_.get_params()}")
train_pred = model.predict(train_X)
train_error = mean_squared_error(train_pred, train_y)
print(f"Training error: {train_error:.4E}.")
test_pred = list()
# batched prediction because it won't fit in memory
for chunk in np.array_split(test_X, 50):
    values = model.predict(chunk)
    test_pred.append(values)
# stack em up
test_pred = np.concatenate(test_pred)
test_error = mean_squared_error(test_pred, test_y)
print(f"Test error: {test_error:.4E}.")
pipeline = make_pipeline(scaler, model)

dump(pipeline, "../models/rate_gp_pipeline.pkl")
#with parallel_backend("threading", n_jobs=1):
#    result = search.fit(X, y)


