from systemssimulator.networks import *
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt

adjacency_matrix = np.array([[0., -1., -1., -1., -1.],
                             [-1., 0., -1., 0., -0.],
                             [-1., -1., 0., -1., 0.],
                             [-1., 0., -1., 0., -1.],
                             [-1., -0., 0., -1., 0.]])

m = np.array([[0.0, 0, 1, 0, 0],
              [0, 1.0 / np.sqrt(2), 0, 1.0 / np.sqrt(2), 0],
              [-1.0 / np.sqrt(2), 0, 0, 0, 1.0 / np.sqrt(2)],
              [1.0 / np.sqrt(2), 0, 0, 0, 1.0 / np.sqrt(2)],
              [0, -1.0 / np.sqrt(2), 0, 1.0 / np.sqrt(2), 0]])

b = np.array([[4. / 3., 1. / 3.],
              [1. / 3., 4. / 3.]])

d = np.array([[1., - np.sqrt(2) / 4., - np.sqrt(2) / 4],
              [- np.sqrt(2) / 3., 2 / 3, -1 / 3],
              [- np.sqrt(2) / 3, -1 / 3, 2 / 3]])

net = Network(nodes=tuple(System() for n in range(5)),
              a=adjacency_matrix)

# weights = np.arange(.125, 2., 0.1)
weights = np.array([2.])
eigsb = np.zeros((len(weights), 2))
eigsd = np.zeros((len(weights), 3))
for i in range(len(weights)):
    net.change_link_strength((1, 4), -weights[i])
    g = inv(m).dot(net._c).dot(m)
    b = g[:2, :2]
    d = g[2:, 2:]
    eigsb[i] = np.sort(eig(b)[0])
    eigsd[i] = np.sort(eig(d)[0])

print(eigsb)
print(eigsd)
# plt.plot(weights, eigsb, '-*', color='b')
# plt.plot(weights, eigsd, '-o', color='r')
# plt.axvline(x=1.0)
# plt.show()
