import numpy as np

A = np.array([
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
])

D = np.diag(np.sum(A, axis=1))
L = D - A

L = np.linalg.inv(D)**0.5 @ L @ np.linalg.inv(D)**0.5
dir = 'data_mibici_2018_4/'
np.save(dir+'laplacian.npy', L)
print('Laplacian matrix saved to', dir+'laplacian.npy') 