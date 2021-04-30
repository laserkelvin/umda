# Unsupervised Molecule Discovery in Astrophysics

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

Work in progress!

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
This version of the cookiecutter template is modified by Kelvin Lee.

