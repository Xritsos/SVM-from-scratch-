import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from matplotlib import pyplot as plt

from data_modules import load
from ploting import plot
from SVM.SVMClassifier import SVMClassifier
from SVM.kernels import linear, rbf, poly, sigmoid
from dim_reduction.KPCA import KPCA
from dim_reduction.LDA import LDA


if __name__ == '__main__':
    # test
    path_train = './datasets/cifar/data_batch_1'
    path_test = './datasets/cifar/data_batch_3'
    
    x_train, y_train = load.get_data(path_train)
    x_train, y_train = load.subsample(x_train, y_train, 3)
    
    x_test, y_test = load.get_data(path_test)
    x_test, y_test = load.subsample(x_test, y_test, 3)
    
    y_train = load.preprocess(y_train, 3)
    
    y_test = load.preprocess(y_test, 3)
    
    scaler = StandardScaler()
    scaler.fit(x_train)
                        
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    kpca = KPCA(n_components=1100, kernel=rbf, gamma=0.0005, 
                coef=None, degree=None)

    kpca.fit(x_train)

    x_train = kpca.transform(x_train)
    x_test = kpca.transform(x_test)

    # LDA
    print()
    print("Running LDA....")
    lda = LDA(n_components=1)
    lda.fit(x_train, y_train)

    x_train = lda.transform(x_train)
    x_test = lda.transform(x_test)

    # SVM
    print()
    print("Started SVM.....")
    clf = SVMClassifier(C=5, kernel=rbf, gamma=0.2, 
                        degree=None, coef=None)
    clf.fit(x_train, np.expand_dims(y_train, axis=1))

    y_pred = clf.predict(x_test)

    np.nan_to_num(y_pred, copy=False, nan=0.0)

    f1_mine = f1_score(y_test, y_pred, average='macro') * 100
    prec_mine = precision_score(y_test, y_pred, average='macro') * 100
    rec_mine = recall_score(y_test, y_pred, average='macro') * 100

    print()
    print("===============")
    print(f"F1 = {f1_mine}")
    print(f"Precision = {prec_mine}")
    print(f"Recall = {rec_mine}")
    print("===============")
    
    
    ################# GRID SEARCH #########################
    # gamma_1 = [0.00001, 0.0001]
    # n_comp = [1000, 800]
    
    # # gamma_2 = [0.01, 0.1, 1]
    # # degree = [5, 3]
    # # coef = [-1, 0, 1]
    # C = [0.1, 1, 10, 100]
    
    # f1s = []
    # precisions = []
    # recalls = []
    # params = []
    
    # for n in n_comp:
    #     for g_1 in gamma_1:
    #         for c in C:
                        
    #             path_train = './datasets/cifar/data_batch_1'
    #             path_val = './datasets/cifar/data_batch_2'
    #             # path_test = './datasets/cifar/data_batch_3'
                
    #             # load train, val and test sets and apply subsampling
    #             x_train, y_train = load.get_data(path_train)
    #             x_train, y_train = load.subsample(x_train, y_train, 3)
                
    #             x_val, y_val = load.get_data(path_val)
    #             x_val, y_val = load.subsample(x_val, y_val, 3)
                
    #             # x_test, y_test = load.get_data(path_test)
    #             # x_test, y_test = load.subsample(x_test, y_test, 3)
                
    #             # fix labels at -1, 1 (1 for target class -1 for the other two)
    #             y_train = load.preprocess(y_train, 3)
    #             y_val = load.preprocess(y_val, 3)
    #             # y_test = load.preprocess(y_test, 3)
                
    #             # scale data
    #             scaler = StandardScaler()
    #             scaler.fit(x_train)
                
    #             x_train = scaler.transform(x_train)
    #             x_val = scaler.transform(x_val)
    #             # x_test = scaler.transform(x_test)
                
    #             print()
    #             print("=========================================")
    #             print(f"n_components: {n}")
    #             print(f"gamma_kpca: {g_1}")
    #             # print(f"gamma_svm: {g_2}")
    #             # print(f"coef: {cf}")
    #             print(f"C: {c}")
    #             print("=========================================")
                
    #             params.append((f'n: {n}', f'gamma_kpca: {g_1}', f'C: {c}'))
                
    #             # KPCA
    #             print()
    #             print("Running KPCA.....")
    #             kpca = KPCA(n_components=n, kernel=rbf, gamma=g_1, 
    #                         coef=None, degree=None)
                
    #             kpca.fit(x_train)
                
    #             x_train = kpca.transform(x_train)
    #             x_val = kpca.transform(x_val)
                
    #             # LDA
    #             print()
    #             print("Running LDA....")
    #             lda = LDA(n_components=1)
    #             lda.fit(x_train, y_train)
                
    #             x_train = lda.transform(x_train)
    #             x_val = lda.transform(x_val)

    #             # SVM
    #             print()
    #             print("Started SVM.....")
    #             clf = SVMClassifier(C=c, kernel=linear, gamma=None, 
    #                                 degree=None, coef=None)
                
    #             clf.fit(x_train, np.expand_dims(y_train, axis=1))

    #             y_pred = clf.predict(x_val)

    #             np.nan_to_num(y_pred, copy=False, nan=0.0)

    #             f1_mine = f1_score(y_val, y_pred, average='macro') * 100
    #             prec_mine = precision_score(y_val, y_pred, average='macro') * 100
    #             rec_mine = recall_score(y_val, y_pred, average='macro') * 100
                
    #             print()
    #             print("===============")
    #             print(f"F1 = {f1_mine}")
    #             print(f"Precision = {prec_mine}")
    #             print(f"Recall = {rec_mine}")
    #             print("===============")
                
                
    #             f1s.append(f1_mine)
    #             precisions.append(prec_mine)
    #             recalls.append(rec_mine)
    
    # max_index = f1s.index(max(f1s))
    
    # best_f1 = f1s[max_index]
    # best_prec = precisions[max_index]
    # best_rec = recalls[max_index]
    # best_params = params[max_index]
    
    # print()
    # print("======= Best Score ==========")
    # print(f"F1 = {best_f1}")
    # print(f"Prec = {best_prec}")
    # print(f"Rec = {best_rec}")
    # print(f"Best Params: {best_params}")
   
    
    fig = plt.figure()
    
    plt.scatter(x_train[y_train==1], 
                y=[i/i * 0.2 for i in range(1, x_train[y_train==1].shape[0]+1)])
    
    plt.scatter(x_train[y_train==-1], 
                y=[i*0 for i in range(x_train[y_train==-1].shape[0])])
    
    plt.show()