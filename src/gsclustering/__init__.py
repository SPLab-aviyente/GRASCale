import numpy as np
import scipy as sp

def _update_Y(Z, Lambda, rho):
    U, _, Vt = np.linalg.svd(Z + (1/rho)*Lambda, full_matrices=False)
    return U@Vt

def _update_L(X, zs, M, Theta, alpha_1, alpha_2, rho):
    n = M.shape[0]

    A = (alpha_1/(2*alpha_1*alpha_2 + rho))*(X@np.diag(zs)@X.T + (rho/alpha_1)*M + (1/alpha_1)*Theta)
    
    rowsum = np.sum(A, axis=1)
    allsum = np.sum(rowsum)
    trace = np.trace(A)

    A -= rowsum/n
    A -= allsum/(n*n*(n-1))
    A += trace/(n*(n-1))
    A -= 1/(n-1)

    A[np.diag_indices(n)] -= ((n-2)/(n*n*(n-1)))*allsum
    A[np.diag_indices(n)] -= ((n-2)/(n*(n-1)))*trace
    A[np.diag_indices(n)] += 1/(n-1) + 1
    
    return A

def _update_Z(Lc, X, L, Y, Lambda, alpha_1, rho, ss, Z0):
    n_signals = X.shape[1]
    k = len(L)

    F = Z0.copy()
    theta_prev = 1

    Q = np.zeros((n_signals, k))
    for s in range(k):
        Q[:, s] = np.diag(X.T@L[s]@X)

    # Keep it for now for debugging, might remove later
    obj = lambda Z: np.trace(Z.T@Lc@Z) + alpha_1*np.trace(Z.T@Q) + \
                    (rho/2)*np.linalg.norm(Z - Y + (1/rho)*Lambda)
    obj_vals = []

    for iter in range(100):
        Z1 = np.maximum(
            0, (1+ss*rho*sp.sparse.eye(n_signals) - 2*ss*Lc)@F - ss*(alpha_1*Q-rho*Y+Lambda)
        )
        theta = (1+np.sqrt(1 + 4*theta_prev**2))/2

        F = Z1 + ((theta - 1)/theta_prev)*(Z1 - Z0)

        Z0 = Z1
        theta_prev = theta

        obj_vals.append(obj(Z1))

    return Z1


def _update_M(L, Theta, rho):
    n = L.shape[0]

    B = L + (1/rho)*Theta
    M = (B+B.T)/2
    M[M > 0] = 0
    M[np.diag_indices(n)] = B[np.diag_indices(n)]

    return M

def _objective(Lc, X, Z, L, alpha_1, alpha_2):
    # returns the value of original objective function
    
    res = np.trace(Z.T@Lc@Z)
    for s, Ls in enumerate(L):
        res += alpha_1*(np.trace(np.diag(Z[:, s])@X.T@L@X) + alpha_2*np.linalg.norm(Ls)**2)
    
    return res

def _lagrangian(Lc, X, Z, Y, L, M, Lambda, Theta, alpha_1, alpha_2, rho):
    # returns the value of augmented lagrangian
    
    res = _objective(Lc, X, Z, L, alpha_1, alpha_2)
    res += np.trace(Lambda.T@(Z - Y)) + 0.5*rho*np.linalg.norm(Z-Y)**2
    for s, Ls in enumerate(L):
        res += np.trace(Theta[s].T@(Ls - M[s])) + 0.5*rho*np.linalg.norm(Ls-M[s])**2

    pass

def cluster_gs(X, Lc, alpha_1, alpha_2, k, rho, max_iter=1000, seed=None, return_obj=False):
    # TODO: Docstring

    n_nodes, n_signals = X.shape

    rng = np.random.default_rng(seed=seed)

    # Init variables
    Z = rng.uniform(size=(n_signals, k))
    Lambda = np.zeros((n_signals, k))
    M = [np.zeros((n_nodes, n_nodes)) for s in range(k)] # Should I start random?
    Theta = [np.zeros((n_nodes, n_nodes)) for s in range(k)]

    # Optimization params
    fista_step_size = 2*np.linalg.norm(Lc, 2) + rho

    if return_obj: # should I return objective and lagrangian values at each iteration?
        objective_vals = []
        lagrangian_vals = []

    # Iterations
    for _ in range(max_iter):
        # Y-step
        Y = _update_Y(Z, Lambda, rho)

        # L-step
        L = [None]*k
        for s in range(k):
            L[s] = _update_L(X, Z[:, s], M[s], Theta[s], alpha_1, alpha_2, rho)

        # Z-step
        Z = _update_Z(Lc, X, L, Y, Lambda, alpha_1, rho, fista_step_size, Z)

        # M-step
        for s in range(k):
            M = _update_M(L[s], Theta[s], rho)
        
        # Update Lagrangian multipliers
        Lambda += rho*(Z - Y)
        for s in range(k):
            Theta[s] += rho*(L[s] - M[s])

        if return_obj:
            objective_vals.append(_objective(Lc, X, Z, L, alpha_1, alpha_2))
            lagrangian_vals.append(_lagrangian(Lc, X, Z, Y, L, M, Lambda, Theta, alpha_1, alpha_2, rho))