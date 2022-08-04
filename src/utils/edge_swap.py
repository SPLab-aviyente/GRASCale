import numpy as np
import networkx as nx

def topological_undirected(G, n_swap=1, seed=None):
    """Perform edge swapping while preserving degree distribution. The method
    randomly select two edges: (a, b, w_ab) and (c, d, w_cd) where w_ab and w_cd
    are edge weights. Then it removes these edges and add (a, d, w_ab) and (c,
    b, w_cd). Edge swapping is done in place. There is a edge swapping function
    in networkx, but it does not handle edge weights. 

    Parameters
    ----------
    G : networkx graph
        An undirected binary/weighted graph
    n_swap : int, optional
        Number of edge swap to perform, by default 1
    """

    n = G.number_of_nodes()
    m = G.number_of_edges()

    is_weighted = nx.is_weighted(G)

    # If the graph is fully connected, there is nothing to randomize
    if m == (n*(n-1)/2):
        return
    
    # Get the nodes of the graph as a list
    node_labels = [i for i in G.nodes]

    rng = np.random.default_rng(seed=seed)

    max_attempt = round(n*m/(n*(n-1)))
    
    for _ in range(n_swap):     

        # Select an edge to swap: ensures that a and b have at least two edges
        attempt = 0
        while attempt <= max_attempt:
            a = rng.choice(node_labels)

            if G.degree(a) > 1:
                b = rng.choice(list(G.neighbors(a)))
            
                if G.degree(b) > 1:
                    break


        # Select two nodes to connect with an edge
        attempt = 0
        while attempt <= max_attempt:
            c = rng.choice(node_labels)
            d = rng.choice(node_labels)

            if c != b and c != a and d!=a and d != b and (not G.has_edge(c, d)):
                break
                
            attempt += 1
        
        # Rewire the edge
        if is_weighted:
            w_ab = G[a][b]['weight']
            G.remove_edge(a, b)
            G.add_edge(c, d, weight=w_ab)
        else:
            G.remove_edge(a, b)
            G.add_edge(a, d)
            