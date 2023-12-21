import pickle
import numpy as np
from numpy import linalg
from scipy.linalg import eigh, eig, qz
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA as KPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from LDA import LDA as MLDA
from PCA import PCA_SVD
from KPCA import KPCA_SVD
from kernels import linear, rbf, sigmoid, poly



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



if __name__ == '__main__':
    path = './datasets/cifar/data_batch_1'
    path_test = './datasets/cifar/test_batch'
    
    batch = unpickle(path)
    test_batch = unpickle(path_test)
    
    data = batch[b'data']
    labels = np.asarray(batch[b'labels'])
    
    test_data = test_batch[b'data']
    test_labels = np.asarray(test_batch[b'labels'])
    
    test_cats = test_data[test_labels==3]
    test_deers = test_data[test_labels==4]
    test_horses = test_data[test_labels==7]
    
    test_cat_y = test_labels[test_labels==3]
    test_deer_y = test_labels[test_labels==4]
    test_horse_y = test_labels[test_labels==7]
    
    x_test = np.concatenate((test_cats, test_deers, test_horses))
    y_test = np.concatenate((test_cat_y, test_deer_y, test_horse_y))
    
    cats = data[labels==3][:900, :]
    deers = data[labels==4][:900, :]
    horses = data[labels==7][:900, :]
    
    cat_y = labels[labels==3][:900]
    deer_y = labels[labels==4][:900]
    horse_y = labels[labels==7][:900]
    
    x = np.concatenate((cats, deers, horses))
    y = np.concatenate((cat_y, deer_y, horse_y))
    
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)
    
    my_pca = KPCA_SVD(n_components=2700, kernel=rbf, gamma=0.00001, degree=50, coef=0)
    my_pca.fit(x)
    x = my_pca.transform(x)
    x_test = my_pca.transform(x_test)
    
    # my_lda = MLDA(n_components=2)
    # my_lda.fit(x, y)
    
    # x_trans_mine = my_lda.transform(x)
    
    lda_skl = LDA(n_components=2)
    lda_skl.fit(x, y)
    
    x_trans_skl = lda_skl.transform(x)
    x_trans_skl = lda_skl.transform(x_test)
    
    # print(x_trans_mine[:4, 0])
    # print(x_trans_skl[:4, 0])
    
    fig = plt.figure()
    
    plt.scatter(x_trans_skl[:, 0][y_test==3], x_trans_skl[:, 1][y_test==3])
    plt.scatter(x_trans_skl[:, 0][y_test==4], x_trans_skl[:, 1][y_test==4])
    plt.scatter(x_trans_skl[:, 0][y_test==7], x_trans_skl[:, 1][y_test==7])
    
    plt.show()
    
    # fig = plt.figure()
    
    # plt.scatter(x_trans_mine[:, 0][y==3], x_trans_mine[:, 1][y==3])
    # plt.scatter(x_trans_mine[:, 0][y==4], x_trans_mine[:, 1][y==4])
    # plt.scatter(x_trans_mine[:, 0][y==7], x_trans_mine[:, 1][y==7])
    
    # plt.show()
    
    
    
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
    