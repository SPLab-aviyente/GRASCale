
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp

from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import spectral_clustering, KMeans
from sklearn import preprocessing
from sklearn.metrics import normalized_mutual_info_score
from scipy import sparse

from src.data import synthetic
from src.gsclustering import cluster_gs_bcd

k = 3
n_nodes = 100
n_signals = [200, 200, 100]
p_cluster = 0.02 # edge probability of ER graphs of the clusters

# Generate data
graphs = []
Xs = []
L = []
gt_clusters = []
for s in range(k):
    G, X = synthetic.gen_smooth_gs_er(n_nodes, p_cluster, n_signals[s])
    graphs.append(G)
    Xs.append(X)
    L.append(nx.laplacian_matrix(G).todense())
    gt_clusters.extend([s]*n_signals[s])

X = np.concatenate(Xs, axis=1)
X_scaled = preprocessing.scale(X, axis=0)

# Create a Gc as a knn graph
Wc = kneighbors_graph(X_scaled.T, 3)
Wc += Wc.T
degrees = np.squeeze(np.asarray(Wc.sum(axis=1)))
Lc = sparse.diags(degrees, 0) - Wc

clusters = spectral_clustering(Wc, n_clusters=k)
print(normalized_mutual_info_score(gt_clusters, clusters))
# e, v = sparse.linalg.eigsh(Lc, k=2, which="SM")
# cl = KMeans(n_clusters=2).fit_predict(v)

# Z, L = cluster_gs(X_scaled, Lc, alpha_1=1, n_clusters=k, rho=1, max_iter=100, L=L)

# print(normalized_mutual_info_score(gt_clusters, KMeans(n_clusters=k).fit_predict(Z)))

Z, L = cluster_gs_bcd(X_scaled, Lc, alpha_1=0.1, n_clusters=k, alpha_2=1)

for s in range(k):
    print(np.count_nonzero(L[s][np.triu_indices_from(L[s], k=1)])/(n_nodes*(n_nodes-1)/2))

print(normalized_mutual_info_score(gt_clusters, KMeans(n_clusters=k).fit_predict(Z)))

# obj, lagr, Z = cluster_gs(X_scaled, Lc, 1, 0.01, 2, 1, return_obj=True, max_iter=50)

# print(Z)

# print(normalized_mutual_info_score(gt_clusters, np.argmax(Z, axis=1)))