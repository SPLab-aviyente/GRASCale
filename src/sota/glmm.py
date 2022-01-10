def run(X, iterations, n_clusters, spread=0.1, regul=0.15, norm_par=1.5):
    
    n_nodes, n_signals = X.shape

    # Initialization