Unsupervised Molecule Discovery in Astrophysics
==============================

Applying cheminformatics to molecular astrophysics

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   │                    predictions
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   ├── presentation   <- Generated graphics and figures to be used in talks
    │   └── writeup        <- Write reports in Markdown and process with Pandoc
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   │── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   │
    │   └── pipeline       <- Scripts that automate the process of preparing,
    │       │                 cleaning, and formatting the project data. This
    │       │                 kind of script is not suited for a notebook, since
    │       │                 it should be run headlessly on a cluster of sorts.
    │       │                 
    │       │                 
    │       ├── main.py    <- This drives the entire pipeline in one script.
    │       ├── make_dataset.py
    │       ├── combine_dataset.py
    │       ├── clean_dataset.py
    │       └── augment_dataset.py
    │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
This version of the cookiecutter template is modified by Kelvin Lee.

# Workflow

This template sets out a modular, pipeline-based workflow for (data) science
projects.


