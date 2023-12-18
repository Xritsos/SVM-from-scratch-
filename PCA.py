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
        n, f = x.shape
        
        min_dim = min(n, f)
        
        if self.n_comp > min_dim:
            raise ValueError("Number of components should be less or equal to" \
                f"the min dimension! That is: {min_dim} !")
        
        self.mean = np.mean(x, axis=0)
        
        x = x - self.mean
        
        u, sigma, v_trans = linalg.svd(x, full_matrices=False)
        
        # the values of the sigma matrix are already in descending order
        # so our eigenvalues do not need sorting 
        
        self.eigen_values = sigma ** 2 / (n - 1)
        self.eigen_vectors = v_trans.T


    def transform(self, x):
        x = x - self.mean
        
        x_transformed = np.dot(x, self.eigen_vectors[:, :self.n_comp])
        
        return x_transformed


    def explained_var(self):
        
        total_variance = np.sum(self.eigen_values)
        
        n_variance = [i/total_variance for i in self.eigen_values[:self.n_comp]]
        
        return n_variance




def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


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
    
    var = my_pca.explained_var()
    
    print()
    print(f"My explained variance: {np.sum(var)}")
    print(f"SKlearn's explained variance: {np.sum(pca_sk.explained_variance_ratio_)}")
    