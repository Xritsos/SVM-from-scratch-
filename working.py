import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.svm import SVC
from svm_module import SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from cvxopt import matrix, solvers


def plot_decision_boundary(X, y, alphas, b, sv, sigma=1):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', s=50)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    xy = np.column_stack((xx.ravel(), yy.ravel()))

    # Evaluate decision function on grid
    decision_function = np.sum(gaussian_kernel(sv, xy, sigma=sigma) * alphas * sv_y, axis=0) + b
    Z = decision_function.reshape(xx.shape)

    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # Plot support vectors
    ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none', edgecolors='k', marker='o')

    plt.show()


def gaussian_kernel(x, z, sigma):
    n = x.shape[0]
    m = z.shape[0]
    xx = np.dot(np.sum(np.power(x, 2), 1).reshape(n, 1), np.ones((1, m)))
    zz = np.dot(np.sum(np.power(z, 2), 1).reshape(m, 1), np.ones((1, n)))     
    return np.exp(-(xx + zz.T - 2 * np.dot(x, z.T)) / (2 * sigma ** 2))


X, y = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=3)
y = np.expand_dims(y, axis=1).astype(float)
y[y == 0] = -1

m, n = X.shape
K = gaussian_kernel(X, X, sigma=1)
P = matrix(np.matmul(y,y.T) * K)
q = matrix(np.ones((m, 1)) * -1)
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros(1))
G = matrix(np.eye(m) * -1)
h = matrix(np.zeros(m))

solution = solvers.qp(P, q, G, h, A, b, kktsolver='ldl')
alphas = np.array(solution['x'])
ind = (alphas > 1e-4).flatten()
sv = X[ind]
sv_y = y[ind]
alphas = alphas[ind]

b = sv_y - np.sum(gaussian_kernel(sv, sv, sigma=1) * alphas * sv_y, axis=0)
b = np.sum(b) / b.size

prod = np.sum(gaussian_kernel(sv, X, sigma=1) * alphas * sv_y, axis=0) + b
predictions = np.sign(prod)

print(predictions, end='')
print(y.T[0])
print(np.where(y.T != predictions))

plot_decision_boundary(X, y, alphas, b, sv)