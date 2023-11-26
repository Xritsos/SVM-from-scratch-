import numpy as np
from cvxopt import matrix, solvers
from kernels import poly, linear, rbf, sigmoid    




def kernel_svm(x, y, x_test, C, gamma):
    n_samples = x.shape[0]
    n_features = x.shape[1]
    
    K = rbf(x, x, gamma=1.0)
    
    H = np.dot(y, y.T) * K
    
    I = np.ones((n_samples), dtype=float) * (-1.0)
    
    A = y.T
    b = 0.0
    
    # second constraint a<=C and -a<=0
    g_1 = np.identity(n_samples, dtype=float)            # first inequality coefficient
    g_2 = np.identity(n_samples, dtype=float) * (-1.0)   # second inequality coefficient
    
    # G a <= h
    G = np.concatenate((g_2, g_1), axis=0)
    
    h_1 = np.ones((n_samples), dtype=float) * C   # first inequality is C
    h_2 = np.zeros((n_samples), dtype=float)           # second inequality is 0
    
    h = np.concatenate((h_2, h_1), axis=0)
    
    P = matrix(H)
    q = matrix(I)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol["x"])
    
    ind = (alphas > 1e-5).flatten()
    sv = x[ind]
    sv_y = y[ind]
    alphas = alphas[ind]

    b = sv_y - np.sum(rbf(sv, sv, gamma=gamma) * alphas * sv_y, axis=0)
    b = np.sum(b) / b.size

    prod = np.sum(rbf(sv, x_test, gamma=gamma) * alphas * sv_y, axis=0) + b
    predictions = np.sign(prod)
    
    return predictions
    