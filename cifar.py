import numpy as np
import time
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_modules import load
from ploting import plot
from SVM.SVMClassifier import SVMClassifier
from SVM.kernels import linear, rbf, poly, sigmoid
from dim_reduction.KPCA import KPCA
from dim_reduction.LDA import LDA



def combine():
    path_train_5 = './datasets/cifar/data_batch_5'
    path_train_4 = './datasets/cifar/data_batch_4'
    path_train_3 = './datasets/cifar/data_batch_3'
    path_train_2 = './datasets/cifar/data_batch_2'
    path_test = './datasets/cifar/test_batch'
    
    x_train_5, y_train_5 = load.get_data(path_train_5)
    x_train_4, y_train_4 = load.get_data(path_train_4)
    x_train_3, y_train_3 = load.get_data(path_train_3)
    x_train_2, y_train_2 = load.get_data(path_train_2)
    
    x_train = np.concatenate((x_train_5, x_train_4, x_train_3, x_train_2))
    y_train = np.concatenate((y_train_5, y_train_4, y_train_3, y_train_2))
    #x_train, y_train = load.get_data(path_train_5)
    
    x_test, y_test = load.get_data(path_test)
    
    print()
    print(f"X train: {x_train.shape}")
    print(f"X test: {x_test.shape}")
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    start = time.time()
    
    kpca = KPCA(n_components=1100, kernel=rbf, gamma=0.0005)
    kpca.fit(x_train)
    
    x_train = kpca.transform(x_train)
    x_test = kpca.transform(x_test)
    
    end = time.time()
    
    print()
    print(f"KPCA: samples - {x_train.shape} time: {end-start} sec.")
    
    start = time.time()
    
    lda = LDA(n_components=2)
    lda.fit(x_train, y_train)
    
    x_train = lda.transform(x_train)
    x_test = lda.transform(x_test)
    
    end = time.time()
    print(f"LDA: samples - {x_train.shape} time: {end-start} sec.")
    
    fig = plt.figure()
    
    plt.scatter(x_train[:, 0][y_train==3], x_train[:, 1][y_train==3])
    plt.scatter(x_train[:, 0][y_train==4], x_train[:, 1][y_train==4])
    plt.scatter(x_train[:, 0][y_train==7], x_train[:, 1][y_train==7])
    
    plt.show()
    
    x_train_d, y_train_d = load.subsample(x_train, y_train, 4)
    x_test_d, y_test_d = load.subsample(x_test, y_test, 4)
    
    x_train_h, y_train_h = load.subsample(x_train, y_train, 7)
    x_test_h, y_test_h = load.subsample(x_test, y_test, 7)
    
    x_train_c, y_train_c = load.subsample(x_train, y_train, 3)
    x_test_c, y_test_c = load.subsample(x_test, y_test, 3)
    
    # plot.plot_balance(y_train_d)
    # plot.plot_balance(y_test_d)
    
    # plot.plot_balance(y_train_h)
    # plot.plot_balance(y_test_h)
    
    # plot.plot_balance(y_train_c)
    # plot.plot_balance(y_test_c)
    
    
    y_train_d = load.preprocess(y_train_d, 4)
    y_test_d = load.preprocess(y_test_d, 4)
    
    y_train_h = load.preprocess(y_train_h, 7)
    y_test_h = load.preprocess(y_test_h, 7)
    
    y_train_c = load.preprocess(y_train_c, 3)
    y_test_c = load.preprocess(y_test_c, 3)
    
    
    print()
    print("Deer Classifier")
    print(x_train_d.shape, x_test_d.shape)
    
    print()
    print("Horse Classifier")
    print(x_train_h.shape, x_test_h.shape)
    
    print()
    print("Cat Classifier")
    print(x_train_c.shape, x_test_c.shape)
    
    
    clf_deer = SVMClassifier(C=5, kernel=rbf, gamma=0.2, degree=None, coef=None)
    
    clf_deer.fit(x_train_d, np.expand_dims(y_train_d, axis=1))
    
    
    clf_horse = SVMClassifier(C=5, kernel=rbf, gamma=0.2, degree=None, coef=None)
    
    clf_horse.fit(x_train_h, np.expand_dims(y_train_h, axis=1))
    
    
    clf_cat = SVMClassifier(C=5, kernel=rbf, gamma=0.2, degree=None, coef=None)
    
    clf_cat.fit(x_train_c, np.expand_dims(y_train_c, axis=1))
    
    
    y_pred_deer = clf_deer.predict(x_test_d)
    y_pred_horse = clf_horse.predict(x_test_h)
    y_pred_cat = clf_cat.predict(x_test_c)
    
    np.nan_to_num(y_pred_deer, copy=False, nan=0.0)
    np.nan_to_num(y_pred_horse, copy=False, nan=0.0)
    np.nan_to_num(y_pred_cat, copy=False, nan=0.0)
    
    f1_deer = f1_score(y_test_d, y_pred_deer, average='macro')
    prec_deer = precision_score(y_test_d, y_pred_deer, average='macro')
    rec_deer = recall_score(y_test_d, y_pred_deer, average='macro')
    
    f1_horse = f1_score(y_test_h, y_pred_horse, average='macro')
    prec_horse = precision_score(y_test_h, y_pred_horse, average='macro')
    rec_horse = recall_score(y_test_h, y_pred_horse, average='macro')
    
    f1_cat = f1_score(y_test_c, y_pred_cat, average='macro')
    prec_cat = precision_score(y_test_c, y_pred_cat, average='macro')
    rec_cat = recall_score(y_test_c, y_pred_cat, average='macro')
    
    
    f1 = [f1_deer, f1_horse, f1_cat]
    classifiers = [4, 7, 3]
    
    
    print()
    print("======== Deer Classifier ========")
    print(f"F1: {f1_deer}")
    print(f"Precision: {prec_deer}")
    print(f"Recall: {rec_deer}")
    print("=================================")
    
    print()
    print("======== Horse Classifier ========")
    print(f"F1: {f1_horse}")
    print(f"Precision: {prec_horse}")
    print(f"Recall: {rec_horse}")
    print("=================================")
    
    print()
    print("======== Cat Classifier ========")
    print(f"F1: {f1_cat}")
    print(f"Precision: {prec_cat}")
    print(f"Recall: {rec_cat}")
    print("=================================")
    
    result = np.zeros((y_test.shape))
    
    deer_pred = clf_deer.predict(x_test)
    horse_pred = clf_horse.predict(x_test)
    cat_pred = clf_cat.predict(x_test)
  
    deer_pred[deer_pred==1] = 4
    horse_pred[horse_pred==1] = 7
    cat_pred[cat_pred==1] = 3
    
    result = deer_pred + horse_pred + cat_pred
    result[result==2] = 4
    result[result==5] = 7
    result[result==1] = 3
    
    for i in range(result.shape[0]):
        
        if result[i] == 10:
            # deer - horse prediction
            index = f1.index(max(f1[0], f1[1]))
            result[i] = classifiers[index]
        elif result[i] == 9:
            # cat - horse
            index = f1.index(max(f1[1], f1[2]))
            result[i] = classifiers[index]
        elif result[i] == 6:
            # deer - cat
            index = f1.index(max(f1[0], f1[2]))
            result[i] = classifiers[index]
        elif result[i] == 14:
            # predict all three
            index = f1.index(max(f1))
            result[i] = classifiers[index]
        elif result[i] == -3:
            # no one predicts
            index = f1.index(min(f1))
            result[i] = classifiers[index]
        
            
    print()
    print(f"Results: {np.unique(result)}")
        
    # final scores
    print()
    print(classification_report(y_test, result))
    
    cm = confusion_matrix(y_test, result, labels=[3, 4, 7])
    
    disp = ConfusionMatrixDisplay(cm, display_labels=['cat', 'deer', 'horse'])
    
    disp.plot()
    
    plt.show()
    
    
    
if __name__ == '__main__':
    
    combine()
   