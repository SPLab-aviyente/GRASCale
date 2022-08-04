import numpy as np
from numba import njit
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

from scipy.sparse import csr_matrix

@njit
def _rowsum_mat_entries(n):
    M = int(n*(n-1)) # number of non-diagonel entries in the matrix
    rows = np.zeros((M, ), dtype=np.int64)
    cols = np.zeros((M, ), dtype=np.int64)
    
    offset_1 = 0 # column offset for block of ones in rows
    offset_2 = 0 # column offset for individual ones in rows
    indx = 0
    
    for row in range(n):
        rows[indx:(indx+n-row-1)] = row
        cols[indx:(indx+n-row-1)] = offset_1 + np.arange(n-row-1)
        
        indx += n-row-1
        offset_1 += n-row-1
        
        if row>0:
            rows[indx:(indx+n-row)] = np.arange(row, n)
            cols[indx:(indx+n-row)] = offset_2 + np.arange(n-row)
            
            indx += n-row
            offset_2 += n-row
    return rows, cols


def rowsum_matrix(n):
    """ Returns a matrix which can be used to find row-sum of a symmetric matrix with zero diagonal
    from its vectorized upper triangular part. 
    For nxn symmetric zero-diagonal matrix A, let a be its M=n(n-1)/2 dimensional vector of upper 
    triangular part. Row-sum matrix P is nxM dimensional matrix such that:
    .. math:: Pa = A1
    where 1 is n dimensional all-one vector.
    Parameters
    ----------
    n : int
        Dimension of the matrix.
    Returns
    -------
    P : sparse matrix
        Matrix to be used in row-sum calculation
    """
    rows, cols = _rowsum_mat_entries(n)
    M = len(rows)
    return csr_matrix((np.ones((M, )), (rows, cols)), shape=(n, int(M/2))) 