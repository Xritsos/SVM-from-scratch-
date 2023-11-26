import numpy as np


x = np.array([[3, 23, 6, 1, 8, 9], [12, 7, 7, 2, 6, 4]])
x = x.reshape((6, 2))

y = np.array([1, 1, -1, 1, -1, -1])
y = y.reshape((6, 1))


print(x.shape)
print(y.shape)

H = np.dot((x*y), (x*y).T)

K = np.dot(x, x.T)
L = np.dot(y, y.T)

P = L * K


print(P)
print(H)

i = np.where(P == H)

print(np.unique(i))
