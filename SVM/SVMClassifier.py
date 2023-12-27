import numpy as np
from cvxopt import matrix, solvers
from kernels import poly, linear, rbf, sigmoid    
import time

class SVMClassifier():
    
    def __init__(self, C=1, kernel=None, degree=None, coef=None, gamma=None, 
                 max_iters=100, rel_tol=1e-6, feas_tol=1e-7):
        
        self.C = float(C)
        self.kernel = kernel
        self.w = 0
        self.beta = 0
        
        self.P = 0
        self.q = 0
        self.A = 0
        self.b = 0
        self.G = 0
        self.h = 0
        self.S = 0
        
        self.supp = 0           # support vectors
        self.supp_labels = 0    # support vectors labels
        self.supp_alphas = 0    # support vectors alpha values
        
        self.degree = degree
        self.coef = coef
        self.gamma = gamma
        
        # initialize kernel
        self.kernel = self.kernel(degree=self.degree, coef=self.coef, gamma=self.gamma)
        
        # optimizer options
        self.max_iters = max_iters
        self.rel_tol = rel_tol
        self.feas_tol = feas_tol
        
    
    def fit(self, x, y):
        
        if not isinstance(x, np.ndarray):
            raise TypeError("Array x is not a numpy ndarray !")
        
        if not isinstance(y, np.ndarray):
            raise TypeError("Array y is not a numpy ndarray !")
        
        if len(x.shape) != 2:
            raise ValueError("Array x should be (num_samples, num_features) !")
        
        if len(y.shape) !=2:
            raise ValueError("Array y should be (samples, 1)")
        
        if y.dtype != 'float':
            raise TypeError("Label values should be floats !")
        
        labels = np.unique(y)
        
        if labels[0] != -1.0 and labels[1] != 1.0 :
            raise ValueError("Values of labels should be 1.0, -1.0 !")
        
        
        print()
        print("Fitting Model ....")
        
        start = time.time()
        
        # start creating the matrices
        n_samples, n_features = x.shape
        
        # Gram Matrix
        K = self.kernel(x, x)
        
        # create matrix of 1/2 H a aT
        H = np.dot(y, y.T) * K
            
        # create matrix of aT 1
        I = np.ones((n_samples), dtype=float) * (-1.0)
    
        # initial constraint yT a = 0
        A = y.T
        b = 0.0
    
        # second constraint a<=C and -a<=0
        g_1 = np.identity(n_samples, dtype=float)            # first inequality coefficient
        g_2 = np.identity(n_samples, dtype=float) * (-1.0)   # second inequality coefficient
        
        # G a <= h
        G = np.concatenate((g_2, g_1), axis=0)
        
        h_1 = np.ones((n_samples), dtype=float) * self.C   # first inequality is C
        h_2 = np.zeros((n_samples), dtype=float)           # second inequality is 0
        
        h = np.concatenate((h_2, h_1), axis=0)
        
        # set to format of cvxopt
        self.P = matrix(H)
        self.q = matrix(I)
        self.A = matrix(A)
        self.b = matrix(b)
        self.G = matrix(G)
        self.h = matrix(h)
        
        options = {'show_progress': False,
                   'max_iters': self.max_iters,
                   'reltol': self.rel_tol,
                   'feastol': self.feas_tol}
        
        sol = solvers.qp(self.P, self.q, self.G, self.h, self.A, self.b,
                         options=options, kktsolver='ldl')
        
        alphas = np.array(sol["x"])

        # hyperplane solution
        self.S = ((alphas > 1e-5) & (alphas < self.C)).flatten()
        
        # support vectors
        self.supp = x[self.S]
        self.supp_labels = y[self.S]
        self.supp_alphas = alphas[self.S]
        
        if self.kernel.__name__ == 'linear':   
            self.w = np.dot((self.supp * self.supp_labels).T, self.supp_alphas)
        
        self.beta = np.mean(self.supp_labels - np.sum(self.kernel(self.supp, self.supp)\
                            * self.supp_alphas * self.supp_labels, axis=0))
        
        print()
        print("Finished fitting !")
        
        finish = time.time()
        
        runtime = round(finish - start, 2)
        
        print()
        print(f"Runtime {runtime} seconds !")
        
        
    def predict(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("Array x is not a numpy ndarray !")
       
        result = np.sum(self.kernel(self.supp, x) * self.supp_alphas * \
                            self.supp_labels, axis=0) + self.beta
        
        return np.sign(result)
    