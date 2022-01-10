from networkx.classes.function import degree
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from scipy import sparse

from src.data import synthetic
from src.gsclustering import cluster_gs

k = 2
n_nodes = 100
n_signals = [200, 200]
p_cluster = 0.1 # edge probability of ER graphs of the clusters

# Generate data
graphs = []
Xs = []
for s in range(k):
    G, X = synthetic.gen_smooth_gs_er(n_nodes, p_cluster, n_signals[s])
    graphs.append(G)
    Xs.append(X)

X = np.concatenate(Xs, axis=1)

# Create a Gc as a knn graph
Wc = kneighbors_graph(X.T, 3)
Wc += Wc.T
degrees = np.squeeze(np.asarray(Wc.sum(axis=1)))
Lc = sparse.diags(degrees, 0) - Wc

obj, lagr = cluster_gs(X, Lc, .01, 10, 2, 100, return_obj=True, max_iter=100)

plt.plot(obj)
plt.plot(lagr)
plt.show()
