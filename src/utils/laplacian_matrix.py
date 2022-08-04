# GRASCale - a python package for simultaneous graph signal clustering and graph 
# learning
# Copyright (C) 2022 Abdullah Karaaslanli <evdilak@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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