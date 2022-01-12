import numpy as np

from ..utils import rowsum_matrix

def _project_hyperplane(v, n):
    """ Project v onto the hyperplane defined by np.sum(v) = -n
    """
    return v - (n + np.sum(v))/(len(v))

def _gl(k, d, alpha1, alpha2, degree_reg="l2"):
    """ Learn smooth graphs using linearized ADMM as described in [1].
    
    References
    ----------
    .. [1] Wang, Xiaolu, et al. "An Efficient Alternating Direction Method for Graph Learning from 
       Smooth Signals." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and 
       Signal Processing (ICASSP). IEEE, 2021.
    """
    n = len(d) # number of nodes
    m = len(k) # number of node pairs

    # Convert k and d to column vectors
    if np.ndim(k) == 1:
        k = k[..., None]

    if np.ndim(d) == 1:
        d = d[..., None]

    # Check if degree regularization is correct-+*
    if degree_reg not in ["lb", "l2"]:
        raise ValueError(
            "The input argument 'degree_reg' must be either 'l2' or 'lb'."
        )

    S = rowsum_matrix(n)

    rho = .1
    mu = 1/(0.9/(rho*(2*(n-1))))

    rng = np.random.default_rng()
    w = _project_hyperplane(rng.uniform(low=0, high=1, size=(n, 1)), -n)
    y = np.zeros((n, 1))
    l = rng.uniform(low=-1, high=0, size=(m, 1))

    primal_res = np.zeros(1000)
    dual_res = np.zeros(1000)
    for iter in range(1000):
        # l-step
        l = np.asarray(S.T@(d - rho*w - y - rho*S@l) - 2*k + mu*l)/(2*alpha2+mu)
        l[l>0] = 0

        # w-step
        w_old = w.copy()
        if degree_reg == "l2":
            w = - np.asarray(rho*S@l + y)/(2*alpha1 + rho)
            w = _project_hyperplane(w, -n)
        elif degree_reg == "lb":
            b = np.asarray(y + rho*S@l)
            w = (- b + np.sqrt(b**2 + 4*rho*alpha1))/(2*rho)

        # update y
        y += rho*np.asarray(S@l + w)

        # Calculate residuals
        primal_res[iter] = np.linalg.norm(rho*S.T@(w-w_old))
        dual_res[iter] = np.linalg.norm(S@l + w)

        if iter > 10:
            if primal_res[iter] < 1e-4 and dual_res[iter] < 1e-4:
                break

    l[l>-1e-4] = 0
    return np.abs(l) # Return adjacency 

def learn_smooth_graph(X, desired_density):
    """Learn an undirected graph from a given data matrix based on smoothness assumption. 

    The method learns the graph by optimizing the graph learning problem proposed in [1]. The 
    optimization is done using linearized ADMM algorithm proposed in [2]

    Parameters
    ----------
    X : ndarray
        Data matrix. Its columns are assumed to be smooth graph signals over the unknown graph.
    desired_density : float
        Desired edge density. The method learns a graph whose edge density is equal to given value.

    Returns
    -------
    W : ndarray
        The adjacency matrix of the learned graph.

    References
    ----------
    .. [1] Dong, Xiaowen, et al. "Learning Laplacian matrix in smooth graph signal representations."
       IEEE Transactions on Signal Processing 64.23 (2016): 6160-6173.
    .. [2] Wang, Xiaolu, et al. "An Efficient Alternating Direction Method for Graph Learning from 
       Smooth Signals." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and 
       Signal Processing (ICASSP). IEEE, 2021.
    """

    # convert XX.T as a vector
    K = X@X.T
    k = K[np.triu_indices_from(K, k=1)]
    d = K[np.diag_indices_from(K)]

    n = len(d) # number of nodes
    m = len(k) # number of node pairs

    # intialize graph learning parameter
    alpha = 1 

    while True:
        w = _gl(k, d, alpha, alpha)

        density = np.nonzero(w)/m

        # check the learned graph has desired density, if not adjust alpha and learn again
        if density - desired_density > 5e-2: # alpha is too big
            alpha *= 1/1.5
        elif density - desired_density < -5e-2: # alpha is too small
            alpha *= 1.5
        else:
            break

    # convert vectorized adjacancy matrix to matrix
    W = np.zeros((n, n))
    W[np.triu_indices_from(W, k=1)] = np.squeeze(w)
    W = (W + W.T)/2

    return W