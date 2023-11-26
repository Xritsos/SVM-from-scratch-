import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from ss_test import SVMSoftMargin

from SVMClassifier import SVMClassifier
from kernels import poly, linear, rbf, sigmoid
from svm_kernel import kernel_svm

def load_dataset():
    
    x, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                                n_redundant=0, n_clusters_per_class=1, 
                                flip_y=0, random_state=0, class_sep=0.2)
    
    
    y = np.expand_dims(y, axis=1) * 1.0
    y[y==0] = -1.0
    
    vals, counts = np.unique(y, return_counts=True)
    
    print()
    print(f"Classes: {counts}")
   
    fig = plt.figure()
   
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel())
   
    plt.show()
   
    return x, y


if __name__ == '__main__':
    
    x, y = load_dataset()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                        random_state=1)
    
    # pred = kernel_svm(x=x_train, y=y_train, x_test=x_test, C=1.0, gamma=1.0)
    
    
    # print(pred.shape)
    # # print(y_test.reshape(-1))
    
    # acc_score = accuracy_score(y_test, pred.reshape((-1)))
    
    # print(acc_score)
    print()
    print(f"x train shape: {x_train.shape}")
    print(f"y train shape: {y_train.shape}")
    
    print()
    print(f"x train dtype: {x_train.dtype}")
    print(f"y train dtype: {y_train.dtype}")
    
    print()
    print(f"x test shape: {x_test.shape}")
    print(f"y test shape: {y_test.shape}")
    
    print()
    print(f"x test dtype: {x_test.dtype}")
    print(f"y test dtype: {y_test.dtype}")
    
    print()
    print(f"Train labels {np.unique(y_train)}")
    print(f"Test labels {np.unique(y_test)}")
    
    # custom classifier
    clf = SVMClassifier(C=1, kernel=linear, degree=2, gamma=1.0, coef=1.0,
                        max_iters=100, rel_tol=1e-6, feas_tol=1e-7)
    
    clf.fit(x_train, y_train)
    
    result = clf.predict(x_test)
    
    print()
    print(f"Mine Result shape: {result.shape}")
    print(f"Mine Result values: {np.unique(result)}")
    print(f"Mine Result dtype: {result.dtype}")

    
    # sklearn classifier
    sk_clf = SVC(C=1, kernel='linear', degree=2, gamma=1.0, coef0=1.0)
    
    sk_clf.fit(x_train, np.ravel(y_train))
    
    result_sklearn = sk_clf.predict(x_test)
    
    print()
    print(f"Sklearn Result shape: {result_sklearn.shape}")
    print(f"Sklearn Result values: {np.unique(result_sklearn)}")
    print(f"Sklearn Result dtype: {result_sklearn.dtype}")
    
    
    acc_skl = accuracy_score(y_test, result_sklearn)
    acc_mine = accuracy_score(y_test, result)
    
    print()
    print(f"Sklearn accuracy: {acc_skl}")
    print(f"Mine accuracy: {acc_mine}")
    
    
    w = clf.w
    b = clf.beta
    
    w_sk = sk_clf.coef_[0]
    b_sk = sk_clf.intercept_
    
    print()
    print(f"Mine w: {w}")
    print(f"SKlearn w: {w_sk}")
    
    print()
    print(f"Mine beta: {b}")
    print(f"SKlearn beta: {b_sk}")
    
    
    x_min = -5
    x_max = 5
    y_min = -5
    y_max = 5
    
    xx = np.linspace(x_min, x_max)
    a = -w[0]/w[1]
    yy = a*xx - (b)/w[1]
    margin = 1 / np.sqrt(np.sum(w**2))
    yy_neg = yy - np.sqrt(1 + a**2) * margin
    yy_pos = yy + np.sqrt(1 + a**2) * margin
    plt.figure(figsize=(8, 8))
    plt.plot(xx, yy, "b-")
    plt.plot(xx, yy_neg, "m--")
    plt.plot(xx, yy_pos, "m--")
    colors = ["steelblue", "orange"]
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), alpha=0.5, 
                cmap=matplotlib.colors.ListedColormap(colors), edgecolors="black")
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
    
    