import cvxopt
import numpy as np



class SVMSoftMargin():

    def __init__(self, C: float):
        self.C = C
        self._w = None
        self._b = None

    def train(self, X: np.ndarray, y: np.ndarray):
        n_samples, _ = X.shape
        # compute inputs for cvxopt solver
        K = (X * y[:, np.newaxis]).T
        P = cvxopt.matrix(K.T.dot(K)) # P has shape n*n
        q = cvxopt.matrix(-1 * np.ones(n_samples)) # q has shape n*1
        G = cvxopt.matrix(np.concatenate((-1*np.identity(n_samples), np.identity(n_samples)), axis=0))
        h = cvxopt.matrix(np.concatenate((np.zeros(n_samples), self.C*np.ones(n_samples)), axis=0))
        A = cvxopt.matrix(1.0 * y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        # solve quadratic programming
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        _lambda = np.ravel(solution['x'])
        # find support vectors
        S = np.where((_lambda > 1e-10) & (_lambda <= self.C))[0]
        self._w = K[:, S].dot(_lambda[S])
        M = np.where((_lambda > 1e-10) & (_lambda < self.C))[0]
        self._b = np.mean(y[M] - X[M, :].dot(self._w))
        
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return +1 for positive class and -1 for negative class.
        """
        # print(f"W: {self._w.shape}")
        # print(f"X: {X.shape}")
        
        results = np.sign(X.dot(self._w) + self._b)
        results[results == 0] = 1
        return results