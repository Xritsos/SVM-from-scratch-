import numpy as np
from numpy import linalg
from scipy.linalg import eigh


class LDA():
    
    def __init__(self, n_components):
        self.n_comp = n_components
        self.eigen_values = None
        self.eigen_vectors = None
        
    
    def fit(self, x, y):
        n_samples, n_features = x.shape
        classes = np.unique(y)
    
        if self.n_comp > (len(classes) - 1):
            raise ValueError(f"Number of components must be max number of classes - 1")
    
        # calculate the mean of all samples
        total_mean = np.mean(x, axis=0)
        total_mean = np.expand_dims(total_mean, axis=1)
    
        # initialize within and between classes matrices
        S_W = np.zeros((n_features, n_features), dtype=np.float32)
        S_B = np.zeros((n_features, n_features), dtype=np.float32)
    
        # calculate for each class
        for c in classes:
            x_c = x[y==c]   # samples of ith class
            x_c_mean = np.mean(x_c, axis=0) # mean of samples of ith class
            x_c_mean = np.expand_dims(x_c_mean, axis=1)
            
            n_c = x_c.shape[0]  # number of samples of ith class
            
            # run over all samples of ith class
            sum_ = 0
            for i in range(x_c.shape[0]):
                x_ci = np.expand_dims(x_c[i, :], axis=1)
                
                sum_ += np.dot((x_ci - x_c_mean), (x_ci - x_c_mean).T)
                
            S_W += sum_ 
            
            S_B += np.dot((x_c_mean - total_mean), (x_c_mean - total_mean).T) * n_c
        
        # solve the general eigenvalue problem
        self.eigen_values, self.eigen_vectors = eigh(S_B, S_W)
        self.eigen_vectors = self.eigen_vectors.T
        
        # sort eigenvalues in descending order
        idxs = np.argsort(self.eigen_values)[::-1]
        
        self.eigen_values = self.eigen_values[idxs]
        self.eigen_vectors = self.eigen_vectors[idxs]
        
            
    def transform(self, x):
        
        x_transformed = np.dot(x, self.eigen_vectors[:self.n_comp].T)
        
        return x_transformed
    
    
    def explained_var(self):
        
        total_variance = np.sum(self.eigen_values)
        
        n_variance = [i/total_variance for i in self.eigen_values[:self.n_comp]]
        
        return n_variance
    