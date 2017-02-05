import numpy as np
from numpy.linalg import eig, inv
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


def change_link_strength(c, link, weight_update):
    i, j = link
    # update off-diagonal entries of self._c.
    c[i, j] += weight_update
    c[j, i] += weight_update

    # update diagonal entries of self._c, since row sum of self._c must be 0.
    c[i, i] -= weight_update
    c[j, j] -= weight_update

    return c


def diagonalize(matrix):
    for i in range(matrix.shape[0]):
        matrix[i] /= matrix[i, i]
    return matrix


def non_diagonalize(matrix):
    for i in range(matrix.shape[0]):
        matrix[i] *= len(matrix[matrix[i] != 0.0]) - 1
    return matrix


a = 1.0 / np.sqrt(2)
m = np.array([[0.0, 0, 1, 0, 0],
              [0, a, 0, a, 0],
              [-a, 0, 0, 0, a],
              [a, 0, 0, 0, a],
              [0, -a, 0, a, 0]])

b = np.array([[4. / 3., 1. / 3.],
              [1. / 3., 4. / 3.]])
a_inv = 1.0 / a
d = np.array([[1., -a_inv / 4., -a_inv / 4],
              [-a_inv / 3., 2 / 3, -1 / 3],
              [-a_inv / 3, -1 / 3, 2 / 3]])

c0 = np.array([[4., -1., -1., -1., -1.],
              [-1., 2., -1., 0., -0.],
              [-1., -1., 3., -1., 0.],
              [-1., 0., -1., 3., -1.],
              [-1., -0., 0., -1., 2.]])

weights = np.arange(.125, 2., 0.1)
eigsb = np.zeros((len(weights), 2))
eigsd = np.zeros((len(weights), 3))
for i in range(len(weights)):
    c = np.copy(c0)
    c = change_link_strength(c, (1, 4), -weights[i])
    c = diagonalize(c)
    g = inv(m).dot(c).dot(m)
    b = g[:2, :2]
    d = g[2:, 2:]
    eigsb[i] = np.sort(eig(b)[0])
    eigsd[i] = np.sort(eig(d)[0])

plt.plot(weights, eigsb, '-*', color='b')
plt.plot(weights, eigsd, '-o', color='r')
plt.axvline(x=1.0)
plt.show()
