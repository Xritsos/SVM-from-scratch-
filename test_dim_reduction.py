import pickle
import numpy as np
from numpy import linalg
from scipy.linalg import eigh, eig, qz
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from LDA import LDA as MLDA
from PCA import PCA_SVD



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



if __name__ == '__main__':
    path = './datasets/cifar/data_batch_1'
    
    batch = unpickle(path)
    
    data = batch[b'data']
    labels = np.asarray(batch[b'labels'])
    
    cats = data[labels==3][:100, :]
    deers = data[labels==4][:100, :]
    horses = data[labels==7][:100, :]
    
    cat_y = labels[labels==3][:100]
    deer_y = labels[labels==4][:100]
    horse_y = labels[labels==7][:100]
    
    x = np.concatenate((cats, deers, horses))
    y = np.concatenate((cat_y, deer_y, horse_y))
    
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)
    
    my_pca = PCA_SVD(n_components=100)
    my_pca.fit(x)
    x = my_pca.transform(x)
    
    my_lda = MLDA(n_components=2)
    my_lda.fit(x, y)
    
    x_trans_mine = my_lda.transform(x)
    
    lda_skl = LDA(n_components=2)
    lda_skl.fit(x, y)
    
    x_trans_skl = lda_skl.transform(x)
    
    print(x_trans_mine[:4, 0])
    print(x_trans_skl[:4, 0])
    
    fig = plt.figure()
    
    plt.scatter(x_trans_skl[:, 0][y==3], x_trans_skl[:, 1][y==3])
    plt.scatter(x_trans_skl[:, 0][y==4], x_trans_skl[:, 1][y==4])
    plt.scatter(x_trans_skl[:, 0][y==7], x_trans_skl[:, 1][y==7])
    
    plt.show()
    
    fig = plt.figure()
    
    plt.scatter(x_trans_mine[:, 0][y==3], x_trans_mine[:, 1][y==3])
    plt.scatter(x_trans_mine[:, 0][y==4], x_trans_mine[:, 1][y==4])
    plt.scatter(x_trans_mine[:, 0][y==7], x_trans_mine[:, 1][y==7])
    
    plt.show()
    
    
    
    # my_pca = PCA_SVD(n_components=3)
    # my_pca.fit(x=data)
    # x_new = my_pca.transform(x=data)
    
    # pca_sk = PCA(3)
    # d_new = pca_sk.fit_transform(temp)
    
    # print(x_new[:5, 2])
    # print()
    # print(d_new[:5, 2])
    
    # var = my_pca.explained_var()
    
    # print()
    # print(f"My explained variance: {np.sum(var)}")
    # print(f"SKlearn's explained variance: {np.sum(pca_sk.explained_variance_ratio_)}")
    