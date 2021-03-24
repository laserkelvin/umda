import numpy as np
import numba
from dask import array as da


@numba.njit(fastmath=True)
def cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the cosine angle between two vectors. This yields a
    similarity metric spanning [-1,1], corresponding to (anti)aligned
    vectors.
    
    This is used primarily as a metric for validating the embeddings
    via molecule arithmetic: check stuff like the similarity scores
    between cyanopolyynes, and reactions like phenyl + cn = benzonitrile.

    Parameters
    ----------
    A, B : np.ndarray
        NumPy 1D arrays, typically for the embeddings

    Returns
    -------
    float
        Cosine similarity between A and B
    """
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


@numba.njit(fastmath=True)
def pairwise_similarity(vectors: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity function.

    Parameters
    ----------
    vectors : np.ndarray
        NumPy 2D array with dimensions [n_samples, n_features].

    Returns
    -------
    np.ndarray
        NumPy 2D array, with each element corresponding to
        the cosine similarity between two rows of the input
        vectors.
    """
    n = len(vectors)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine_similarity(vectors[i], vectors[j])
    return matrix


# These are definitions from KIDA; calculate a rate coefficient
# given alpha, beta, gamma, and a temperature
def arrhenius_rate(T: float, alpha: float, beta: float, gamma: float) -> float:
    return alpha * (T / 300.)**beta * np.exp(-gamma/T)


def ionpol1(T: float, alpha: float, beta: float, gamma: float) -> float:
    return alpha * beta * (0.62 + 0.4767 * gamma * (300. / T)**0.5)


def ionpol2(T: float, alpha: float, beta: float, gamma: float) -> float:
    return alpha * beta * (1. + 0.0967 * gamma * (300 / T)**0.5 + (gamma**2. / 10.526) * (300. / T))


def compute_rate(react_class: int, T: float, alpha: float, beta: float, gamma: float) -> float:
    # this provides a mapping function that returns the rate coefficient
    # given a reaction identifier.
    if react_class == 3:
        func = arrhenius_rate
    elif react_class == 4:
        func = ionpol1
    elif react_class == 5:
        func = ionpol2
    else:
        raise NotImplementedError("Reaction class not recgonized.")
    return func(T, alpha, beta, gamma)
