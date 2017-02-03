from systemssimulator.networks import *
import matplotlib.pyplot as plt


net = Network()
net.run()

res1 = net.nodes[0].read_buffer('_state_buffer')[:, 0]
res2 = net.nodes[1].read_buffer('_state_buffer')[:, 0]

plt.figure()
plt.plot(res1)
plt.figure()
plt.plot(res2)
plt.figure()
plt.plot(np.abs(res1 - res2))
plt.show()
