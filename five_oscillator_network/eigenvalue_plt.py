from systemssimulator.networks import *
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt

a = np.array([[1., -1., -1., -1., -1.],
              [-1., 0., -1., 0., -0.],
              [-1., -1., 0., -1., 0.],
              [-1., 0., -1., 0., -1.],
              [-1., -0., 0., -1., 0.]])


def coupling_from_adjacency(matrix):
    if not np.allclose(matrix.diagonal(), np.zeros(matrix.shape[0])):
        raise ValueError('Matrix must have all-zero diagonal')
    for i in range(matrix.shape[0]):
        matrix[i, i] = -sum(matrix[i])
    return matrix

print(coupling_from_adjacency(a))

# c0 = np.array([[4., -1., -1., -1., -1.],
#                [-1., 2., -1., 0., -0.],
#                [-1., -1., 3., -1., 0.],
#                [-1., 0., -1., 3., -1.],
#                [-1., -0., 0., -1., 2.]])
#
# m = np.array([[0.0, 0, 1, 0, 0],
#               [0, 1.0 / np.sqrt(2), 0, 1.0 / np.sqrt(2), 0],
#               [-1.0 / np.sqrt(2), 0, 0, 0, 1.0 / np.sqrt(2)],
#               [1.0 / np.sqrt(2), 0, 0, 0, 1.0 / np.sqrt(2)],
#               [0, -1.0 / np.sqrt(2), 0, 1.0 / np.sqrt(2), 0]])
#
# b = np.array([[4. / 3., 1. / 3.],
#               [1. / 3., 4. / 3.]])
#
# d = np.array([[1., - np.sqrt(2) / 4., - np.sqrt(2) / 4],
#               [- np.sqrt(2) / 3., 2 / 3, -1 / 3],
#               [- np.sqrt(2) / 3, -1 / 3, 2 / 3]])
#
# net = Network(nodes=tuple(System() for n in range(5)),
#               eps=9.6,
#               c=c0)
#
# weights = np.arange(.125, 2., 0.1)
# eigsb = np.zeros((len(weights), 2))
# eigsd = np.zeros((len(weights), 3))
# for i in range(len(weights)):
#     net.c = np.copy(c0)
#     net.change_link_strength((1, 4), -weights[i])
#     net.diagonalize_c()
#     g = inv(m).dot(net.c).dot(m)
#     b = g[:2, :2]
#     d = g[2:, 2:]
#     eigsb[i] = np.sort(eig(b)[0])
#     eigsd[i] = np.sort(eig(d)[0])
#
# plt.plot(weights, eigsb, '-*', color='b')
# plt.plot(weights, eigsd, '-o', color='r')
# plt.axvline(x=1.0)
# plt.show()
