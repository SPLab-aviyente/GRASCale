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
import scipy as sp

from numba import njit

from src.utils import rowsum_matrix

@njit
def _project_to_neg_simplex_bs(v, a, stop_th=1e-2):
    mu = np.min(v) - a

    obj = np.sum(np.minimum(v - mu, 0)) - a

    while np.abs(obj) > stop_th:
        obj = np.sum(np.minimum(v - mu, 0)) - a
        df = - np.sum((v - mu) < 0)
        mu -= obj/df

    return mu

def mat2simplex(matX, l=1.):
    # TODO: Copy-pasted from some github gist, need reference
    matX = matX.T
    m, n = matX.shape
    matS = np.sort(matX, axis=0)[::-1]
    matC = np.cumsum(matS, axis=0) - l
    matH = matS - matC / (np.arange(m) + 1).reshape(m, 1)
    matH[matH<=0] = np.inf
    r = np.argmin(matH, axis=0)
    t = matC[r,np.arange(n)] / (r + 1)
    matY=matX-t 
    matY[matY<0] = 0
    return matY.T

def _random_laplacian(m, n):
    rng = np.random.default_rng()
    l = rng.normal(size=(m, 1))
    l[l>0] = 0
    return l/np.abs(np.sum(l))*n

def _objective(Lc, X, Z, L, alpha_1, alpha_2, S):
    # returns the value of original objective function
    n_nodes = X.shape[0]
    n_clusters = len(L)

    res = np.trace(Z.T@Lc@Z)
    for s in range(n_clusters):
    
        res += alpha_1*np.trace(sp.sparse.diags(Z[:, s], 0)@(X.T@L[s]@X)) 
        res += alpha_1*alpha_2*np.sum(Z[:, s])*(L[s].power(2).sum())
    
    return res

def run_init(X, Lc, n_clusters, alpha_1, alpha_2, b=10, max_iter=100, seed=None):

    n_nodes, n_signals = X.shape
    n_pairs = n_nodes*(n_nodes-1)//2

    S = rowsum_matrix(n_nodes)

    
    rng = np.random.default_rng(seed=seed)

    # Optimization params
    spectral_norm_Lc, _ = sp.sparse.linalg.eigsh(Lc, k=1)
    spectral_norm_Lc = spectral_norm_Lc.item()
    step_size_z = 2*spectral_norm_Lc # Lipschitz constant of gradient of f wrt Z

    step_size_l = 4*alpha_2*n_nodes # Lipschitz constant of gradient of f wrt ls

    w = 0.8 # Extrapolation weight

    triu_rows, triu_cols = np.triu_indices(n_nodes, k=1)

    Zs = []
    for b_iter in range(b):

        # Init variables
        Z = np.zeros((n_signals, n_clusters))
        Z[(np.arange(n_signals), np.argmax(rng.uniform(size=(n_signals, n_clusters)), axis=1))] = 1

        Z_prev = np.zeros((n_signals, n_clusters))
        Z_prev[(np.arange(n_signals), np.argmax(rng.uniform(size=(n_signals, n_clusters)), axis=1))] = 1

        l = []
        l_prev = []

        for i in range(n_clusters):
            l.append(_random_laplacian(n_pairs, n_nodes))
            l_prev.append(np.zeros((n_pairs, 1)))
        
        # BCD Steps
        for iter in range(max_iter):

            ## Update Ls
            for s in range(n_clusters):
                K = X@sp.sparse.diags(Z[:, s], 0)@X.T/np.sum(Z[:, s])
                k = K[np.triu_indices_from(K, k=1)][..., None]
                d = K[np.diag_indices_from(K)][..., None]

                l_hat = (1+w)*l[s] - w*l_prev[s]
                g = 2*k - S.T@d + 4*alpha_2*l_hat + 2*alpha_2*S.T@(S@l_hat)
                v = l_hat - g/step_size_l

                l_prev[s] = l[s]

                mu = _project_to_neg_simplex_bs(v, -n_nodes)
                l[s] = v - mu
                l[s][l[s] > 0] = 0

            ## Update Z
            Q = np.zeros((n_signals, n_clusters)) # Data matrix for smoothness term
            Q2 = np.ones((n_signals, n_clusters))
            L = []
            for s in range(n_clusters):
                # convert vectorized laplacian to matrix form
                data = np.concatenate((l[s][l[s]!=0], l[s][l[s]!=0], -np.squeeze(S@l[s])))
                rows = np.concatenate((triu_rows[np.squeeze(l[s])!=0], 
                                    triu_cols[np.squeeze(l[s])!=0], 
                                    np.arange(n_nodes)))
                cols = np.concatenate((triu_cols[np.squeeze(l[s])!=0], 
                                    triu_rows[np.squeeze(l[s])!=0], 
                                    np.arange(n_nodes)))
                L.append(sp.sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)))

                Q2[:, s] = np.sum(data**2)
                Q[:, s] = np.diag(X.T@L[s]@X)
                
            Z_hat = (1+w)*Z - w*Z_prev
            G = 2*Lc@Z_hat + alpha_1*Q + alpha_1*alpha_2*Q2
            V = Z_hat - G/step_size_z

            Z_prev = Z

            Z = mat2simplex(V)
         
        Zs.append(Z)
    
    return Zs

def run(X, Lc, n_clusters, alpha_1, alpha_2, max_iter=200, Z=None, seed=None):
    n_nodes, n_signals = X.shape
    n_pairs = n_nodes*(n_nodes-1)//2

    S = rowsum_matrix(n_nodes)

    # Init variables
    rng = np.random.default_rng(seed=seed)

    if Z is None:
        Z = np.zeros((n_signals, n_clusters))
        Z[(np.arange(n_signals), np.argmax(rng.uniform(size=(n_signals, n_clusters)), axis=1))] = 1

    Z_prev = np.zeros((n_signals, n_clusters))

    l = []
    l_prev = []
    for i in range(n_clusters):
        l.append(_random_laplacian(n_pairs, n_nodes))
        l_prev.append(np.zeros((n_pairs, 1)))

    # Optimization params
    spectral_norm_Lc, _ = sp.sparse.linalg.eigsh(Lc, k=1)
    spectral_norm_Lc = spectral_norm_Lc.item()
    step_size_z = 2*spectral_norm_Lc # Lipschitz constant of gradient of f wrt Z

    step_size_l = 4*alpha_2*n_nodes # Lipschitz constant of gradient of f wrt ls

    w = 0.8 # Extrapolation weight

    objective_vals = []

    triu_rows, triu_cols = np.triu_indices(n_nodes, k=1)

    for iter in range(max_iter):
        ## Update Ls
        for s in range(n_clusters):
            K = X@sp.sparse.diags(Z[:, s], 0)@X.T/np.sum(Z[:, s])
            k = K[np.triu_indices_from(K, k=1)][..., None]
            d = K[np.diag_indices_from(K)][..., None]

            l_hat = (1+w)*l[s] - w*l_prev[s]
            g = 2*k - S.T@d + 4*alpha_2*l_hat + 2*alpha_2*S.T@(S@l_hat)
            v = l_hat - g/step_size_l

            l_prev[s] = l[s]

            mu = _project_to_neg_simplex_bs(v, -n_nodes)
            l[s] = v - mu
            l[s][l[s] > 0] = 0

        ## Update Z
        Q = np.zeros((n_signals, n_clusters)) # Data matrix for smoothness term
        Q2 = np.ones((n_signals, n_clusters))
        L = []
        for s in range(n_clusters):
            # convert vectorized laplacian to matrix form
            data = np.concatenate((l[s][l[s]!=0], l[s][l[s]!=0], -np.squeeze(S@l[s])))
            rows = np.concatenate((triu_rows[np.squeeze(l[s])!=0], 
                                   triu_cols[np.squeeze(l[s])!=0], 
                                   np.arange(n_nodes)))
            cols = np.concatenate((triu_cols[np.squeeze(l[s])!=0], 
                                   triu_rows[np.squeeze(l[s])!=0], 
                                   np.arange(n_nodes)))
            L.append(sp.sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)))

            Q2[:, s] = np.sum(data**2)
            Q[:, s] = np.diag(X.T@L[s]@X)
            
        Z_hat = (1+w)*Z - w*Z_prev
        G = 2*Lc@Z_hat + alpha_1*Q + alpha_1*alpha_2*Q2
        V = Z_hat - G/step_size_z

        Z_prev = Z

        Z = mat2simplex(V)
    
        objective_vals.append(_objective(Lc, X, Z, L, alpha_1, alpha_2, S))

        if iter > 10 and np.abs(objective_vals[iter]/objective_vals[iter-1] - 1) < 1e-6:
           break

    # Convert vectorized laplacian to adjacency matrices
    W = []
    for s in range(n_clusters):
        W.append(np.zeros((n_nodes, n_nodes)))
        W[s][np.triu_indices_from(W[s], k=1)] = -np.squeeze(l[s])
        W[s] = (W[s] + W[s].T)/2
    
    return Z, W, objective_vals[-1]