from systemssimulator.networks import *
import matplotlib.pyplot as plt


net = Network(eps=9.6)
t_run = 50.
while net._network_t_clock <= t_run:
            net(delta_t=default.t_step)
            net._network_t_clock += default.t_step
            if np.isclose(net._network_t_clock, 0.25 * t_run):
                print(net._c)
                net.remove_link((0, 1))
                print(net._c)
            if np.isclose(net._network_t_clock, 0.75 * t_run):
                print(net._c)
                net.add_link((0, 1))
                print(net._c)

x0 = net.nodes[0].read_buffer('_state_buffer')[:, 0]
x1 = net.nodes[1].read_buffer('_state_buffer')[:, 0]

avg = (x0 + x1) / net._node_num
delta = np.var(x0 - avg)
plt.plot((x0 - avg) / delta)
plt.plot((x1 - avg) / delta)
plt.show()



