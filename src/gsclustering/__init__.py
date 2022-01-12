import numpy as np
import scipy as sp

from scipy.sparse import csr_matrix
from numba import njit

def _update_Y(Z, Lambda, rho):
    U, _, Vt = np.linalg.svd(Z + (1/rho)*Lambda, full_matrices=False)
    return U@Vt

def _project_to_H4(A):
    n = A.shape[0]

    rowsum = np.sum(A, axis=1)[..., None]
    allsum = np.sum(rowsum)
    trace = np.trace(A)

    A -= rowsum/n
    A -= allsum/(n*n*(n-1))
    A += trace/(n*(n-1))
    A -= 1/(n-1)

    A[np.diag_indices(n)] += (n/(n*n*(n-1)))*allsum
    A[np.diag_indices(n)] -= (n/(n*(n-1)))*trace
    A[np.diag_indices(n)] += 1/(n-1) + 1
    
    return A

def _update_L(X, zs, M, Theta, alpha_1, alpha_2, rho):

    c = alpha_1/(2*alpha_1*alpha_2 + rho)
    A = -c*(X@np.diag(zs)@X.T - (rho/alpha_1)*M + (1/alpha_1)*Theta)
    
    return _project_to_H4(A)


def _project_to_simplex(V, a=1):
    # TODO: ref: mblondel/projection_simplex.py github
    n_signals, k = V.shape
    for i in range(n_signals):
        v = V[i, :]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - a
        ind = np.arange(k) + 1
        cond = u - cssv/ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1]/float(rho)
        V[i, :] = np.maximum(0, v-theta)

    return V       
    
def _update_Z(Lc, X, L, alpha_1, ss, Z0, Y=None, Lambda=None, rho=None):
    n_signals = X.shape[1]
    k = len(L)

    # Data matrix for smoothness term
    Q = np.zeros((n_signals, k))
    for s in range(k):
        Q[:, s] = np.diag(X.T@L[s]@X)

    # FISTA initialization
    F = Z0.copy()
    theta_prev = 1

    # is the problem constrained with orthogonality?
    orthogonality = True
    if rho is None:
        orthogonality = False

    # objective function
    if orthogonality:
        obj_func = lambda Z: np.trace(Z.T@Lc@Z) + alpha_1*np.trace(Z.T@Q) + \
                             (rho/2)*np.linalg.norm(Z - Y + (1/rho)*Lambda)**2
    else:
        obj_func = lambda Z: np.trace(Z.T@Lc@Z) + alpha_1*np.trace(Z.T@Q)

    obj_vals = []

    for iter in range(1000):
        
        # Z1 = np.maximum(
        #     0, ((1 - ss*rho)*sp.sparse.eye(n_signals) - 2*ss*Lc)@F - ss*(alpha_1*Q-rho*Y+Lambda)
        # )

        if orthogonality:
            Z1 = _project_to_simplex(
                ((1 - ss*rho)*sp.sparse.eye(n_signals) - 2*ss*Lc)@F - ss*(alpha_1*Q-rho*Y+Lambda)
            )
        else:
            Z1 = _project_to_simplex(
                (sp.sparse.eye(n_signals) - 2*ss*Lc)@F - ss*alpha_1*Q
            )

        # update FISTA parameter
        theta = (1+np.sqrt(1 + 4*theta_prev**2))/2

        # update accelaration variable
        F = Z1 + ((theta - 1)/theta_prev)*(Z1 - Z0)

        Z0 = Z1
        theta_prev = theta

        obj_vals.append(obj_func(Z1))

        # if objective value is not changing, terminate FISTA
        if (iter > 5) and np.abs(obj_vals[iter] - obj_vals[iter-1]) < 1e-4:
            break
        
    return Z1

def _project_to_H3(B):
    n = B.shape[0]

    M = (B+B.T)/2
    M[M > 0] = 0
    M[np.diag_indices(n)] = B[np.diag_indices(n)]
    return M

def _update_M(L, Theta, rho):

    B = L + (1/rho)*Theta

    return _project_to_H3(B)

def _objective(Lc, X, Z, L, alpha_1, alpha_2):
    # returns the value of original objective function
    
    res = np.trace(Z.T@Lc@Z)
    for s, Ls in enumerate(L):
        res += alpha_1*(np.trace(np.diag(Z[:, s])@X.T@Ls@X) + alpha_2*np.linalg.norm(Ls)**2)
    
    return res

def _lagrangian(Lc, X, Z, Y, L, M, Lambda, Theta, alpha_1, alpha_2, rho):
    # returns the value of augmented lagrangian
    
    res = _objective(Lc, X, Z, L, alpha_1, alpha_2)
    res += np.trace(Lambda.T@(Z - Y)) + 0.5*rho*np.linalg.norm(Z-Y)**2
    for s, Ls in enumerate(L):
        res += np.trace(Theta[s].T@(Ls - M[s])) + 0.5*rho*np.linalg.norm(Ls-M[s])**2

    return res

def cluster_gs(X, Lc, alpha_1, n_clusters, alpha_2=None, rho=None, max_iter=1000, L=None, orthogonality=True, 
               seed=None):
    # TODO: Docstring

    n_nodes, n_signals = X.shape

    learn_graphs = True
    if L is not None:
        learn_graphs = False

    # Init variables
    rng = np.random.default_rng(seed=seed)
    Z = rng.uniform(size=(n_signals, n_clusters))
    if orthogonality:
        Lambda = np.zeros((n_signals, n_clusters))   

    if learn_graphs:
        M = [np.zeros((n_nodes, n_nodes)) for s in range(n_clusters)] # Should I start random?
        Theta = [np.zeros((n_nodes, n_nodes)) for s in range(n_clusters)]

    # Optimization params
    spectral_norm_Lc, _ = sp.sparse.linalg.eigsh(Lc, k=1)
    spectral_norm_Lc = spectral_norm_Lc.item()
    if orthogonality:
        fista_step_size = 1/(2*spectral_norm_Lc + rho)
    else:
        fista_step_size = 1/(2*spectral_norm_Lc)

    # Iterations
    for iter in range(max_iter):
        
        if orthogonality:
            # Y-step
            Y = _update_Y(Z, Lambda, rho)

        # L-step
        if learn_graphs:
            L = [None]*n_clusters
            for s in range(n_clusters):
                L[s] = _update_L(X, Z[:, s], M[s], Theta[s], alpha_1, alpha_2, rho)
        # print(_objective(Lc, X, Z, L, alpha_1, alpha_2))

        # Z-step
        if orthogonality:
            Z = _update_Z(Lc, X, L, alpha_1, fista_step_size, Z, Y, Lambda, rho)
        else:
            Z = _update_Z(Lc, X, L, alpha_1, fista_step_size, Z)
            if not learn_graphs:
                return Z

        # print(_objective(Lc, X, Z, L, alpha_1, alpha_2))

        # M-step
        if learn_graphs:
            for s in range(n_clusters):
                M[s] = _update_M(L[s], Theta[s], rho)
        
        # Update Lagrangian multipliers
        if orthogonality:
            Lambda += rho*(Z - Y)

        if learn_graphs:
            for s in range(n_clusters):
                Theta[s] += rho*(L[s] - M[s])
                # print("{}".format(np.linalg.norm(L[s]-M[s])), end=" ")
            # print()

        # print(_lagrangian(Lc, X, Z, Y, L, M, Lambda, Theta, alpha_1, alpha_2, rho))
        print(_objective(Lc, X, Z, L, alpha_1, alpha_2))
            
    return Z, M

def _random_laplacian(m, n):
    rng = np.random.default_rng()
    l = rng.normal(size=(m, 1))
    l[l>0] = 0
    return l/np.abs(np.sum(l))*n

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

@njit
def _project_to_neg_simplex_bs(v, a, stop_th=1e-2):
    mu = np.min(v) - a

    obj = np.sum(np.minimum(v - mu, 0)) - a

    while np.abs(obj) > stop_th:
        obj = np.sum(np.minimum(v - mu, 0)) - a
        df = - np.sum((v - mu) < 0)
        mu -= obj/df

    return mu

def _objective_bcd(Lc, X, Z, l, alpha_1, alpha_2, S):
    # returns the value of original objective function
    n_nodes = X.shape[0]
    n_clusters = len(l)

    res = np.trace(Z.T@Lc@Z)
    for s in range(n_clusters):
        # convert vectorized laplacian to matrix form
        L = np.zeros((n_nodes, n_nodes))
        L[np.triu_indices_from(L, k=1)] = np.squeeze(l[s])
        L = L + L.T
        L[np.diag_indices_from(L)] = -np.squeeze(S@l[s])
    
        res += alpha_1*(np.trace(np.diag(Z[:, s])@X.T@L@X) + alpha_2*np.linalg.norm(L)**2)
    
    return res

def _project_to_steifel(Z):
    U, _, Vt = np.linalg.svd(Z, full_matrices=False)
    return U@Vt

def cluster_gs_bcd(X, Lc, n_clusters, alpha_1, alpha_2, max_iter=500, seed=None):
    n_nodes, n_signals = X.shape
    n_pairs = n_nodes*(n_nodes-1)//2

    S = rowsum_matrix(n_nodes)

    # Init variables
    rng = np.random.default_rng(seed=seed)

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

    w = .8 # Extrapolation weight

    for iter in range(max_iter):
        ## Update Ls
        for s in range(n_clusters):
            K = X@np.diag(Z[:, s])@X.T
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
        for s in range(n_clusters):
            # convert vectorized laplacian to matrix form
            L = np.zeros((n_nodes, n_nodes))
            L[np.triu_indices_from(L, k=1)] = np.squeeze(l[s])
            L = L + L.T
            L[np.diag_indices_from(L)] = -np.squeeze(S@l[s])
            Q[:, s] = np.diag(X.T@L@X)

        Z_hat = (1+w)*Z - w*Z_prev
        G = 2*Lc@Z_hat + alpha_1*Q
        V = Z_hat - G/step_size_z

        Z_prev = Z

        Z = _project_to_steifel(V)

        print(_objective_bcd(Lc, X, Z, l, alpha_1, alpha_2, S))

    # Convert vectorized laplacian to adjacency matrices
    W = []
    for s in range(n_clusters):
        W.append(np.zeros((n_nodes, n_nodes)))
        W[s][np.triu_indices_from(W[s], k=1)] = -np.squeeze(l[s])
        W[s] = (W[s] + W[s].T)/2
    
    return Z, W