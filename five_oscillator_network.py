from systemssimulator.networks import *
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt

adjacency_matrix = np.array([[0., -1., -1., -1., -1.],
                             [-1., 0., -1., 0., -1.],
                             [-1., -1., 0., -1., 0.],
                             [-1., 0., -1., 0., -1.],
                             [-1., -1., 0., -1., 0.]])

a = 1.0 / np.sqrt(2)
m = np.array([[0., 0, 1, 0, 0],
              [0, a, 0, a, 0],
              [-a, 0, 0, 0, a],
              [a, 0, 0, 0, a],
              [0, -a, 0, a, 0]])

net = Network(nodes=tuple(System() for n in range(5)),
              a=adjacency_matrix,
              eps=3.)

t_run = 250.
while net._network_t_clock <= t_run:
    net(delta_t=default.t_step)
    net._network_t_clock += default.t_step
    if np.isclose(net._network_t_clock, np.array([50.0])):
        print(net._c)
        net.change_link_strength((1, 4), (-2.0))
        # net.remove_link((0, 2))
        print(net._c)

weights = np.arange(0., 2., 0.1)
eigsb = np.zeros((len(weights), 2))
eigsd = np.zeros((len(weights), 3))
for i in range(len(weights)):
    net.change_link_strength((1, 4), -weights[i])
    g = inv(m).dot(net._c).dot(m)
    b = g[:2, :2]
    d = g[2:, 2:]
    eigsb[i] = np.sort(eig(b)[0])
    eigsd[i] = np.sort(eig(d)[0])

plt.plot(weights, net.eps * eigsb, '-*', color='b')
plt.plot(weights, net.eps * eigsd, '-o', color='r')
plt.axvline(x=1.0)
plt.axhline(y=10.0)
plt.xlabel(r'$w_{2,5}$', fontsize=16)
plt.ylabel(r'$\varepsilon \lambda_i$', fontsize=16)


fig, ax = plt.subplots(4)
ax[0].plot(np.abs(net.nodes[0].read_buffer('_state_buffer')[:, 0] - net.nodes[1].read_buffer('_state_buffer')[:, 0]))
ax[0].set_ylabel(r'$|x_{} - x_{}|$'.format(0, 1), fontsize=16)
ax[1].plot(np.abs(net.nodes[1].read_buffer('_state_buffer')[:, 0] - net.nodes[4].read_buffer('_state_buffer')[:, 0]))
ax[1].set_ylabel(r'$|x_{} - x_{}|$'.format(2, 5), fontsize=16)
ax[2].plot(np.abs(net.nodes[2].read_buffer('_state_buffer')[:, 0] - net.nodes[3].read_buffer('_state_buffer')[:, 0]))
ax[2].set_ylabel(r'$|x_{} - x_{}|$'.format(3, 4), fontsize=16)
ax[3].plot(np.abs(net.nodes[0].read_buffer('_state_buffer')[:, 0] - net.nodes[4].read_buffer('_state_buffer')[:, 0]))
ax[3].set_ylabel(r'$|x_{} - x_{}|$'.format(1, 5), fontsize=16)
plt.tight_layout()
plt.show()
