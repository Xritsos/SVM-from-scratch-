import numpy as np
from numpy import linalg


class PCA_SVD():
    
    def __init__(self, n_components):
        
        self.n_comp = n_components
        self.eigen_vectors = None
        self.eigen_values = None
        self.mean = 0
        
        
    def fit(self, x):
        n, f = x.shape
        
        min_dim = min(n, f)
        
        if self.n_comp > min_dim:
            raise ValueError("Number of components should be less or equal to " \
                f"the min dimension! That is: {min_dim} !")
        
        # first center the data
        self.mean = np.mean(x, axis=0)
        x = x - self.mean
        
        u, sigma, v_trans = linalg.svd(x, full_matrices=False)
        
        # the values of the sigma matrix are already in descending order
        # so our eigenvalues do not need sorting 
        
        self.eigen_values = sigma ** 2 / n
        
        # the values of sigma are the eigenvalues for the A.T A matrix
        # we need the eigenvalues for the A matrix
        # that is the square of the sigmas
        
        self.eigen_vectors = v_trans.T


    def transform(self, x):
        # center the data 
        x = x - self.mean
        
        x_transformed = np.dot(x, self.eigen_vectors[:, :self.n_comp])
        
        return x_transformed


    def explained_var(self):
        
        total_variance = np.sum(self.eigen_values)
        
        n_variance = [i/total_variance for i in self.eigen_values[:self.n_comp]]
        
        return n_variance
