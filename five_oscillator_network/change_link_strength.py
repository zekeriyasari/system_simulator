from systemssimulator.networks import *
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt

adjacency_matrix = np.array([[0., -1., -1., -1., -1.],
                             [-1., 0., -1., 0., -0.125],
                             [-1., -1., 0., -1., 0.],
                             [-1., 0., -1., 0., -1.],
                             [-1., -0.125, 0., -1., 0.]])

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
              eps=9.6,
              a=adjacency_matrix)
print(net._c)
g = inv(m).dot(net._c).dot(m)
b = g[:2, :2]
d = g[2:, 2:]
b_eigs = np.sort(eig(b)[0])
d_eigs = np.sort(eig(d)[0])
print(b_eigs)
print(d_eigs)
print(b_eigs[0] * net.eps)
print(d_eigs[1] * net.eps)
# net.change_link_strength((1, 4), weight=-2.)
# print(net._c)
# g = inv(m).dot(net._c).dot(m)
# b = g[:2, :2]
# d = g[2:, 2:]
# b_eigs = np.sort(eig(b)[0])
# d_eigs = np.sort(eig(d)[0])
# print(b_eigs)
# print(d_eigs)
# print(b_eigs[0] * net.eps)
# print(d_eigs[1] * net.eps)
t_run = 50.
while net._network_t_clock <= t_run:
            net(delta_t=default.t_step)
            net._network_t_clock += default.t_step
            if np.isclose(net._network_t_clock, 25.):
                print(net._c)
                g = inv(m).dot(net._c).dot(m)
                b = g[:2, :2]
                d = g[2:, 2:]
                b_eigs = np.sort(eig(b)[0])
                d_eigs = np.sort(eig(d)[0])
                print(b_eigs)
                print(d_eigs)
                print(b_eigs[0] * net.eps)
                print(d_eigs[1] * net.eps)
                net.change_link_strength((1, 4), weight=-20.)
                print(net._c)
                g = inv(m).dot(net._c).dot(m)
                b = g[:2, :2]
                d = g[2:, 2:]
                b_eigs = np.sort(eig(b)[0])
                d_eigs = np.sort(eig(d)[0])
                print(b_eigs)
                print(d_eigs)
                print(b_eigs[0] * net.eps)
                print(d_eigs[1] * net.eps)

fig, ax = plt.subplots(4)
ax[0].plot(np.abs(net.nodes[0].read_buffer('_state_buffer')[:, 0] - net.nodes[1].read_buffer('_state_buffer')[:, 0]))
ax[1].plot(np.abs(net.nodes[1].read_buffer('_state_buffer')[:, 0] - net.nodes[4].read_buffer('_state_buffer')[:, 0]))
ax[2].plot(np.abs(net.nodes[2].read_buffer('_state_buffer')[:, 0] - net.nodes[3].read_buffer('_state_buffer')[:, 0]))
ax[3].plot(np.abs(net.nodes[3].read_buffer('_state_buffer')[:, 0] - net.nodes[4].read_buffer('_state_buffer')[:, 0]))
plt.show()
