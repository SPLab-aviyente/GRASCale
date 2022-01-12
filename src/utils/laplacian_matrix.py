import numpy as np

from scipy import sparse

def laplacian_matrix(A):
    """Return (unnormalized) Laplacian matrix for the given adjacency matrix.

    Parameters
    ----------
    A : ndarray or sparse matrix
        Adjacency matrix

    Returns
    -------
    L : ndarray or sparse matrix
        The Laplacian matrix. If A is a sparse matrix, the L is a sparse matrix. If A is a numpy 
        array, L is a numpy array.
    """

    degrees = np.squeeze(np.asarray(A.sum(axis=1)))
    if sparse.issparse(A):
        L = sparse.diags(degrees, 0) - A
    else:
        L = np.diag(degrees) - A

    return L 