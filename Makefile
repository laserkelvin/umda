
packages=ipython numba numpy rdkit scikit-learn matplotlib dask dask-ml jupyter h5py

environment:
	conda create -n umda -c conda-forge $(packages)
	conda activate umda
	pip install -r requirements.txt
