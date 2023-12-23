import numpy as np
from numpy import linalg

from PCA import PCA_SVD


class KPCA():
    
    def __init__(self, n_components, kernel, degree=None, coef=None, gamma=None):
        self.n_comp = n_components
        self.eigen_values = None
        self.eigen_vectors = None
        self.K = None
        self.X_fit = None
        self.kernel = kernel
        self.degree = degree
        self.coef = coef
        self.gamma = gamma
        
        # initialize kernel
        self.kernel = self.kernel(degree=self.degree, coef=self.coef, gamma=self.gamma)
        
        self.mean = 0
        
    def fit(self, x):
        self.X_fit = x
        
        n_samples, n_features = x.shape
        
        min_dim = min(n_samples, n_features)
        
        if self.n_comp > min_dim:
            raise ValueError("Number of components should be less or equal to " \
                f"the min dimension! That is: {min_dim} !")
        
        
        if self.kernel.__name__ == 'linear':
            pca = PCA_SVD(n_components=self.n_comp)
            pca.fit(x)
            
            self.eigen_values = pca.eigen_values
            self.eigen_vectors = pca.eigen_vectors
            self.mean = pca.mean
        else:
            # apply kernel
            self.K = self.kernel(x, x)
            
            I = np.ones((self.K.shape))
        
            # center Gram Matrix
            self.K = self.K - (1/n_samples) * np.dot(I, self.K) - (1/n_samples) * \
                        np.dot(self.K, I) + (1/n_samples**2) * np.dot(np.dot(I, self.K), I)
            
            # eigen decomposition
            self.eigen_values, self.eigen_vectors = linalg.eigh(self.K)
        
            idxs = np.argsort(self.eigen_values)[::-1]
            
            self.eigen_values[idxs]
            self.eigen_vectors[:, idxs]
        
        
    def transform(self, x):
        if self.kernel.__name__ == 'linear':
            x = x - self.mean
            x_transformed = np.dot(x, self.eigen_vectors[:, :self.n_comp])
        else:
            self.K = self.kernel(x, self.X_fit)
            print(f"K matrix: {self.K.shape}")
            x_transformed = np.dot(self.K, self.eigen_vectors[:, :self.n_comp])
        
        return x_transformed
        