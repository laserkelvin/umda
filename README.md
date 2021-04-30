# Unsupervised Molecule Discovery in Astrophysics

![umap-image](umap_image.png)

## Applying cheminformatics to astrochemistry

This repository includes notebooks and codebase for developing machine learning
pipelines that apply cheminformatics concepts to predicting astrochemical properties.

The current focus is on molecular column densities in astronomical observations,
but can potentially be applied towards laboratory data, as well as studying chemical
networks. As it stands, the code has been tested to work for up to four million
molecules on a Dell XPS 15 (32 GB ram, 6 core i7-9750H) without much difficulty
thanks to frameworks like `dask` that can abstract away a large amount of the
parallelization and out-of-memory operations.

## Installation

Currently, the codebase is not quite ready for public consumption: while the
API more or less works as intended, there's still a bit of fussing around with
model training and deploying. If you would like to contribute to this aspect,
please raise an issue in this repository!

The `Makefile` `environment` recipe should recreate the software environment
needed for `umda` to work. Simply run `make environment` to set everything
up automatically.

## Instructions

Currently a user API is underdeveloped, and so if you would like to run your
own predictions it is somewhat manual. As part of the repository, we've included
a pretrained embedding model, as well as a host of regressors stored as `pickle`s
dumped using `joblib`.

Here is an example of the bare minimum code one needs to run the model and
predict the column density of benzene and formaldehyde using linear regression:

```python
from joblib import load
import numpy as np

# load a wrapper class for generating embeddings
embedder = load("models/EmbeddingModel.pkl")
regressor = load("models/linear_regression_grid.pkl").best_estimator_

smiles = ["C1=C=C=C=C=C1", "C=O"]
vecs = np.vstack([embedder.vectorize(smi) for smi in smiles])
regressor.predict(vecs)
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
This version of the cookiecutter template is modified by Kelvin Lee.

