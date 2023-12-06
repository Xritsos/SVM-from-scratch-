import numpy as np
import pickle
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.svm import SVC

from SVMClassifier import SVMClassifier
from kernels import poly, linear, rbf, sigmoid


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def plot_image(array):
    
    red = array[:1024].reshape((32, 32))
    green = array[1024:2048].reshape((32, 32))
    blue = array[2048:].reshape((32, 32))
    
    rgb = np.dstack((red, green, blue))
    
    fig = plt.figure()
    
    plt.imshow(rgb)
    
    plt.show()
    
    
def plot_balance(labels):
    vals, counts = np.unique(labels, return_counts=True)
    vals = vals.astype(np.uint8)
    
    path_meta = './datasets/cifar/batches.meta'
    
    meta = unpickle(path_meta)
    label_names = meta[b'label_names']
    
    names = [label_names[i].decode("utf-8") for i in vals]
   
    fig, ax = plt.subplots()

    colors = ['red', 'brown', 'orange', 'blue']
    
    ax.bar(names, counts, color=colors, label=names)

    ax.set_ylabel('Counts')
    ax.set_xlabel('Classes')
    ax.set_title('Class Balance')
    ax.legend(title='Targets')
    
    plt.show()
    
    
def get_data(path):
    
    batch = unpickle(path)
    
    data = batch[b'data']
    labels = np.asarray(batch[b'labels'])

    # we choose only the classes deer horse and cat
    # deer --> 4, horse --> 7, cat --> 3
    
    deer = data[labels==4, :]
    horse = data[labels==7, :]
    cat = data[labels==3, :]
    
    new_data = np.concatenate((deer, horse, cat))
    
    deer_lb = labels[labels==4]
    horse_lb = labels[labels==7]
    cat_lb = labels[labels==3]
    
    new_labels = np.concatenate((deer_lb, horse_lb, cat_lb))
    
    return new_data, new_labels


def preprocess(y_train, y_test, class_id):
    # the targer class each time will be assigned to 1
    y_train[y_train==class_id] = 1
    y_train[y_train!=1] = -1
    
    y_train = y_train.astype(np.float64)
    
    print()
    print(f"Train labels: {np.unique(y_train)}")
    
    y_test[y_test==class_id] = 1
    y_test[y_test!=1] = -1
    
    y_test = y_test.astype(np.float64)
    
    print()
    print(f"Test labels: {np.unique(y_test)}")
    
    
    return y_train, y_test


def subsample(x_train, x_test, y_train, y_test, class_id):
    classes = [3, 4, 7]
    classes.remove(class_id)
    
    target_tr = y_train==class_id
    target_te = y_test==class_id
    
    max_sample_tr = math.floor(x_train[target_tr, :].shape[0] / 2)
    max_sample_te = math.floor(x_test[target_te, :].shape[0] / 2)
    
    other_1_tr = y_train==classes[0]
    other_2_tr = y_train==classes[1]
    
    other_1_te = y_test==classes[0]
    other_2_te = y_test==classes[1]
    
    x_train_class = x_train[target_tr, :]
    x_test_class = x_test[target_te, :]
    y_train_class = y_train[target_tr]
    y_test_class = y_test[target_te]
    
    # train
    x_train_other_1 = x_train[other_1_tr, :]
    y_train_other_1 = y_train[other_1_tr]
    
    rand = np.random.randint(0, x_train_other_1.shape[0], max_sample_tr)
    
    x_train_other_1 = x_train_other_1[rand, :]
    y_train_other_1 = y_train_other_1[rand]
    
    x_train_other_2 = x_train[other_2_tr, :]
    y_train_other_2 = y_train[other_2_tr]
    
    rand = np.random.randint(0, x_train_other_2.shape[0], max_sample_tr)
    
    x_train_other_2 = x_train_other_2[rand, :]
    y_train_other_2 = y_train_other_2[rand]
    
    x_train_new = np.concatenate((x_train_class,
                                  x_train_other_1,
                                  x_train_other_2))
    
    y_train_new = np.concatenate((y_train_class,
                                  y_train_other_1,
                                  y_train_other_2))
    
    # test
    x_test_other_1 = x_test[other_1_te, :]
    y_test_other_1 = y_test[other_1_te]
    
    rand = np.random.randint(0, x_test_other_1.shape[0], max_sample_te)
    
    x_test_other_1 = x_test_other_1[rand, :]
    y_test_other_1 = y_test_other_1[rand]
    
    x_test_other_2 = x_test[other_2_te, :]
    y_test_other_2 = y_test[other_2_te]
    
    rand = np.random.randint(0, x_test_other_2.shape[0], max_sample_te)
    
    x_test_other_2 = x_test_other_2[rand, :]
    y_test_other_2 = y_test_other_2[rand]
    
    x_test_new = np.concatenate((x_test_class,
                                  x_test_other_1,
                                  x_test_other_2))
    
    y_test_new = np.concatenate((y_test_class,
                                  y_test_other_1,
                                  y_test_other_2))
    
    
    return x_train_new, x_test_new, y_train_new, y_test_new



def deer_classifier(C, kernel, gamma, degree, coef, rel_tol, feas_tol):
    path_train = './datasets/cifar/data_batch_1'
    path_test = './datasets/cifar/data_batch_5'
    
    # get data (deer, horse, cat) from each train batch and test batch
    x_train, y_train = get_data(path_train)
    x_test, y_test = get_data(path_test)
    
    # as it takes time we will only take some samples out of each class and 
    # also create class balance for the 'one vs all'
    x_train, x_test, y_train, y_test = subsample(x_train, x_test, y_train, y_test, 4)
    
    # plot_balance(y_train)
    # plot_balance(y_test)
    
    x_train, x_test, y_train, y_test = preprocess(x_train, x_test, y_train, y_test, 4)
    
    print()
    print(f"Train set: {(x_train.shape, y_train.shape)}")
    print(f"Test set: {(x_test.shape, y_test.shape)}")
    
    clf = SVMClassifier(C=C, kernel=kernel, gamma=gamma, degree=degree, coef=coef,
                        rel_tol=rel_tol, feas_tol=feas_tol)
    
    clf.fit(x_train, np.expand_dims(y_train, axis=1))
    
    y_pred = clf.predict(x_test)
    
    np.nan_to_num(y_pred, copy=False, nan=0.0)

    f1_mine = f1_score(y_test, y_pred, average='macro')
    prec_mine = precision_score(y_test, y_pred, average='macro')
    rec_mine = recall_score(y_test, y_pred, average='macro')
    
    return clf, f1_mine, prec_mine, rec_mine
    
    
def horse_classifier(C, kernel, gamma, degree, coef, rel_tol, feas_tol):
    path_train = './datasets/cifar/data_batch_2'
    path_test = './datasets/cifar/data_batch_5'
    
    # get data (deer, horse, cat) from each train batch and test batch
    x_train, y_train = get_data(path_train)
    x_test, y_test = get_data(path_test)
    
    # as it takes time we will only take some samples out of each class and 
    # also create class balance for the 'one vs all'
    x_train, x_test, y_train, y_test = subsample(x_train, x_test, y_train, y_test, 7)
    
    # plot_balance(y_train)
    # plot_balance(y_test)
    
    
    x_train, x_test, y_train, y_test = preprocess(x_train, x_test, y_train, y_test, 7)
    
    print()
    print(f"Train set: {(x_train.shape, y_train.shape)}")
    print(f"Test set: {(x_test.shape, y_test.shape)}")
    
    clf = SVMClassifier(C=C, kernel=kernel, gamma=gamma, degree=degree, coef=coef,
                        rel_tol=rel_tol, feas_tol=feas_tol)
    
    clf.fit(x_train, np.expand_dims(y_train, axis=1))
    
    y_pred = clf.predict(x_test)
    
    np.nan_to_num(y_pred, copy=False, nan=0.0)
    
    f1_mine = f1_score(y_test, y_pred, average='macro')
    prec_mine = precision_score(y_test, y_pred, average='macro')
    rec_mine = recall_score(y_test, y_pred, average='macro')
    
    return clf, f1_mine, prec_mine, rec_mine


def cat_classifier(C, kernel, gamma, degree, coef, rel_tol, feas_tol):
    
    path_train = './datasets/cifar/data_batch_3'
    path_test = './datasets/cifar/data_batch_5'
    
    # get data (deer, horse, cat) from each train batch and test batch
    x_train, y_train = get_data(path_train)
    x_test, y_test = get_data(path_test)
    
    # as it takes time we will only take some samples out of each class and 
    # also create class balance for the 'one vs all'
    x_train, x_test, y_train, y_test = subsample(x_train, x_test, y_train, y_test, 3)
    
    # plot_balance(y_train)
    # plot_balance(y_test)
    
    x_train, x_test, y_train, y_test = preprocess(x_train, x_test, y_train, y_test, 3)
    
    print()
    print(f"Train set: {(x_train.shape, y_train.shape)}")
    print(f"Test set: {(x_test.shape, y_test.shape)}")
    
    clf = SVMClassifier(C=C, kernel=kernel, gamma=gamma, degree=degree, coef=coef,
                        rel_tol=rel_tol, feas_tol=feas_tol)
    
    clf.fit(x_train, np.expand_dims(y_train, axis=1))
    
    y_pred = clf.predict(x_test)
    
    np.nan_to_num(y_pred, copy=False, nan=0.0)
    
    f1_mine = f1_score(y_test, y_pred, average='macro')
    prec_mine = precision_score(y_test, y_pred, average='macro')
    rec_mine = recall_score(y_test, y_pred, average='macro')
    
    
    return clf, f1_mine, prec_mine, rec_mine


def combine():
    path_train = './datasets/cifar/data_batch_5'
    path_test = './datasets/cifar/data_batch_4'
    
    x_train, y_train = get_data(path_train)
    x_test, y_test = get_data(path_test)
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    print()
    print(f"X train: {x_train.shape}")
    print(f"X test: {x_test.shape}")
    
    x_train_d, x_test_d, y_train_d, y_test_d = subsample(x_train, x_test, 
                                                         y_train, y_test, 4)
    
    x_train_h, x_test_h, y_train_h, y_test_h = subsample(x_train, x_test,
                                                         y_train, y_test, 7)
    
    x_train_c, x_test_c, y_train_c, y_test_c = subsample(x_train, x_test, 
                                                         y_train, y_test, 3)
    
    y_train_d, y_test_d = preprocess(y_train_d, y_test_d, 4)
    
    y_train_h, y_test_h = preprocess(y_train_h, y_test_h, 7)
    
    y_train_c, y_test_c = preprocess(y_train_c, y_test_c, 3)
    
    print()
    print("Deer Classifier")
    print(x_train_d.shape, x_test_d.shape)
    
    print()
    print("Horse Classifier")
    print(x_train_h.shape, x_test_h.shape)
    
    print()
    print("Cat Classifier")
    print(x_train_c.shape, x_test_c.shape)
    
    clf_deer = SVMClassifier(C=100, kernel=rbf, gamma=0.001, degree=1, coef=0,
                            rel_tol=1e-6, feas_tol=1e-7)
    
    clf_deer.fit(x_train_d, np.expand_dims(y_train_d, axis=1))
    
    
    clf_horse = SVMClassifier(C=100, kernel=rbf, gamma=0.0007, degree=1, coef=0,
                            rel_tol=1e-6, feas_tol=1e-7)
    
    clf_horse.fit(x_train_h, np.expand_dims(y_train_h, axis=1))
    
    
    clf_cat = SVMClassifier(C=110, kernel=rbf, gamma=0.001, degree=1, coef=0,
                            rel_tol=1e-6, feas_tol=1e-7)
    
    clf_cat.fit(x_train_c, np.expand_dims(y_train_c, axis=1))
    
    # test each individual scores
    
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
    
    
    # Combine and test them in whole
    
    # plot_balance(y_train)
    
    result = np.zeros((y_test.shape))
    
    deer_pred = clf_deer.predict(x_test)
    horse_pred = clf_horse.predict(x_test)
    cat_pred = clf_cat.predict(x_test)
    
    for i in range(result.shape[0]):
        
        if deer_pred[i] == 1 and horse_pred[i] == 1:
            index = f1.index(max(f1[0], f1[1]))
            result[i] = classifiers[index]
        elif deer_pred[i] == 1 and cat_pred[i] == 1:
            index = f1.index(max(f1[0], f1[2]))
            result[i] = classifiers[index]
        elif cat_pred[i] == 1 and horse_pred[i] == 1:
            index = f1.index(max(f1[1], f1[2]))
            result[i] = classifiers[index]
        elif deer_pred[i] == 1:
            result[i] = 4
        elif horse_pred[i] == 1:
            result[i] = 7
        elif cat_pred[i] == 1:
            result[i] = 3
        else:
            index = f1.index(min(f1))
            result[i] = classifiers[index]
            
    print()
    print(f"Results: {np.unique(result)}")
        
    # final scores
    print()
    print(classification_report(y_test, result))
    
    
if __name__ == '__main__':
    
    combine()
   