from systemssimulator.networks import *
import matplotlib.pyplot as plt


net = Network(eps=default.epsh)
t_run = 50.
while net._network_t_clock <= t_run:
            net(delta_t=default.t_step)
            net._network_t_clock += default.t_step
            if np.isclose(net._network_t_clock, 0.25 * t_run):
                print(net.c)
                net.remove_link((0, 1))
                print(net.c)
            if np.isclose(net._network_t_clock, 0.75 * t_run):
                print(net.c)
                net.add_link((0, 1), 1.0)
                print(net.c)

output0 = net.nodes[0].read_buffer('_output_buffer')
output1 = net.nodes[1].read_buffer('_output_buffer')

plt.figure()
plt.plot(output0)
plt.figure()
plt.plot(output1)
plt.figure()
plt.plot(np.abs(output1 - output0))
plt.show()

