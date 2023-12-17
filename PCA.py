import pickle
import numpy as np
from numpy import linalg
from sklearn.decomposition import PCA


class PCA_SVD():
    
    def __init__(self, n_components):
        
        self.n_comp = n_components
        self.eigen_vectors = None
        self.eigen_values = None
        self.mean = 0
        
        
    def fit(self, x):
        n = x.shape[0]
        
        self.mean = np.mean(x, axis=0)
        
        x = x - self.mean
        
        u, sigma, v_trans = linalg.svd(x, full_matrices=False)
        
        self.eigen_values = sigma ** 2 / (n - 1)
        self.eigen_vectors = v_trans.T
        
        # sort based on max variance (max eigen_values)
        idxs = np.argsort(self.eigen_values)[::-1]
        
        self.eigen_values = self.eigen_values[idxs]
        self.eigen_vectors = self.eigen_vectors[:, idxs]


    def transform(self, x):
        x = x - self.mean
        
        x_transformed = np.dot(x, self.eigen_vectors[:, :self.n_comp])
        
        return x_transformed






def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def pca(x, n_components):
    n = x.shape[0]
    
    mean = np.mean(x, axis=0)
    
    x = x - mean
    
    u, sigma, v_trans = linalg.svd(x, full_matrices=False)
    
    eigen_values = sigma**2 / (n-1)
    
    idxs = np.argsort(eigen_values)[::-1]
    
    eigen_values = eigen_values[idxs]
    eigen_vectors = v_trans.T[:, idxs]
    
    x_transformed = np.dot(x, eigen_vectors[:, :n_components])
    
    return x_transformed


if __name__ == '__main__':
    path = './datasets/cifar/data_batch_1'
    
    batch = unpickle(path)
    
    data = batch[b'data'][:100, :]
    temp = np.copy(data)
    
    my_pca = PCA_SVD(n_components=3)
    my_pca.fit(x=data)
    x_new = my_pca.transform(x=data)
    
    pca_sk = PCA(3)
    d_new = pca_sk.fit_transform(temp)
    
    print(x_new[:5, 2])
    print()
    print(d_new[:5, 2])
    