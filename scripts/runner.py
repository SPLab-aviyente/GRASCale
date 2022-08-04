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

from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import spectral_clustering, KMeans

import project_path
from src.utils import laplacian_matrix
from src import grascale

def run_grascale(X, alpha1, alpha2, n_clusters):
    n_nodes, n_signals = X.shape

    # Generate Gc with knn graph
    Wc = kneighbors_graph(X.T, 5)
    Wc += Wc.T
    Wc[Wc>0] = 1
    
    Lc = laplacian_matrix(Wc)

    ##### INITIALIZATION STEP #####

    # Number of times to run algorithm for initialization
    n_burnin = 9

    Zs = grascale.run_init(X, Lc, n_clusters=n_clusters, alpha_1=alpha1, 
                           alpha_2=alpha2, b=n_burnin)

    # Consensus clustering: Probably need a faster algo
    ZZ = np.zeros((n_signals, n_signals))
    for rj in range(n_burnin):
        ZZ += Zs[rj]@Zs[rj].T

    ZZ[np.diag_indices_from(ZZ)] = 0
    
    clusters = spectral_clustering(ZZ, n_clusters=n_clusters)
    Z0 = np.zeros(Zs[0].shape)
    Z0[(np.arange(n_signals), clusters)]=1

    # Run algorithm with found initial point
    Z, W_hat, _ = grascale.run(X, Lc, alpha_1=alpha1, n_clusters=n_clusters, 
                               alpha_2=alpha2, max_iter=1000, Z = Z0)

    # Find clusters from Z
    cl_hat = KMeans(n_clusters=n_clusters).fit_predict(Z)

    return cl_hat, W_hat