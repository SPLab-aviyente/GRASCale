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

import warnings

import networkx as nx
import numpy as np

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import f1_score

import project_path
from src.utils import edge_swap
from src import data
from runner import run_grascale

warnings.simplefilter(action="ignore", category=FutureWarning)

##### GENERATE DATA #####

# Random model to use to generate base graph and its parameter
graph_generator = nx.erdos_renyi_graph
graph_param = 0.1

n_nodes = 50 # number of nodes in the graphs associated with clusters
n_clusters = 3
n_signals = [200, 200, 200] # number of signals in each cluster

# amount of the perturbation for generating graphs associated with clusters 
perturbation = 1 

w_gt = [] # ground-truth graphs
cl_gt = [] # ground-truth clustering

Xs = []

# Generate a base, ensure that it is a connected graph
i = 1
while True:
    G = graph_generator(n_nodes, graph_param, seed=32 + i)
    if nx.is_connected(G):
        break
    else:
        i += 1

for s in range(n_clusters):

    # Generate graphs associated with clusters from base graph by edge perturbation
    # We ensure that after perturbation graph remains connected
    Gs = G.copy()
    n_swaps = int(Gs.number_of_edges()*perturbation)

    i = 1
    while True:
        edge_swap.topological_undirected(Gs, n_swap=n_swaps, seed=32+s + 30*i)
        if nx.is_connected(Gs):
            break
        else:
            i += 1

    # Generate data
    X = data.gen_smooth_gs(Gs, n_signals[s], seed=32+73*s, filter="Gaussian", noise_amount=0.1)
    Xs.append(X)

    # Save ground truth clustering and graphs
    cl_gt.extend([s]*n_signals[s])
    w_gt.append(nx.to_numpy_array(Gs)[np.triu_indices(n_nodes, k=1)])

X = np.concatenate(Xs, axis=1)

##### RUN GRASCale and REPORT PERFORMANCES #####

alpha1 = 10
alpha2 = 0.35 # this should return graphs whose densities are around 0.14-0.15
cl_hat, W_hat = run_grascale(X, alpha1, alpha2, n_clusters)

print(f"NMI: {nmi(cl_gt, cl_hat):.4f}")

densities = 0
for s in range(n_clusters):
    w_hat = W_hat[s][np.triu_indices(n_nodes, k=1)]
    w_hat[w_hat>0] = 1
    densities += np.count_nonzero(w_hat)/(n_clusters*n_nodes*(n_nodes-1)/2)

print(f"Density: {densities:.4f}")

f1 = 0
for si in range(n_clusters):
    f1s_cluster = []
    for sj in range(n_clusters):
        w_hat = W_hat[sj][np.triu_indices(n_nodes, k=1)]
        w_hat[w_hat>0] = 1
        f1s_cluster.append(f1_score(w_gt[si], w_hat))
    f1 += (np.max(f1s_cluster))/n_clusters

print(f"F1: {f1:.4f}")