"""
combine_dataset.py

This module contains the routines to put all of the data together into
a single output.

What's included here is a cookiecutter template for combining many
files into a single array with pseudocode.

Ideally, this will all turn it into HDF5 format which is much more
portable and managable than just NumPy memmap arrays for large datasets.

Requires Python >= 3.2
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def read_datafiles(filepath):
    """
    Example function for reading in data form a file.
    This needs to be adjusted for the specific format that
    will work for the project.
    
    Parameters
    ----------
    filepath : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    with open(filepath) as read_file:
        filecontents = read_file.read()
    return filecontents


def read_directory(dir_path, pattern="*", max_workers=None):
    """
    Function that will aggregate data from a directory using a pool
    of processors. This pattern is ideally suited for many different
    files where threading is not so great.
    
    By default, the max_workers argument is set to None, which will
    use as many cores as possible.
    
    Parameters
    ----------
    dir_path : str
        A str corresponding to the directory path containing all
        the data files.
    pattern : str, optional
        File matching pattern in for globbing, by default "*"
    max_workers : int or None, optional
        Passed to the `ProcessPoolExecutor` object, setting the maximum
        number of workers; by default None
    
    Returns
    -------
    Combined output of `read_datafiles` function
    """
    dir_path = Path(dir_path)
    with ProcessPoolExecutor(max_workers) as executor:
        for file in dir_path.rglob(pattern):
            # Arguments are passed into the read_datafiles function
            # as a tuple
            contents = executor.map(read_datafiles, (file))
    return contents
