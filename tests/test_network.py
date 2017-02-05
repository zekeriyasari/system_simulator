from unittest import TestCase
from systemssimulator.networks import *
from systemssimulator.systems import get_x_state


class TestNetwork(TestCase):
    def test_check_parameter_type(self):
        print('\nTest: {}'.format(check_parameter_type.__name__))
        int_param = 1
        float_param = 1.0
        tuple_param = (1, 2, 3)
        self.failUnless(check_parameter_type(int_param, int))
        self.failUnless(check_parameter_type(float_param, float))
        self.failUnless(check_parameter_type(tuple_param, tuple))
        with self.assertRaises(TypeError):
            check_parameter_type(int_param, tuple)
        self.failUnless(int_param, (int, numbers.Number))
        print('ok.')

    def test_init(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format(Network.__init__.__name__))
        net = Network()
        for i in range(2):
            self.failUnless(isinstance(net.nodes[i], System))
        self.failUnless(np.allclose(net._c, np.array([[-1.0, 1.0], [1.0, -1.0]])))
        self.failUnless(np.allclose(net.p, np.diag([1, 0, 0])))
        self.failUnless(net.eps == default.epsh)
        self.failUnless(net._node_dim == net.nodes[0]._dim)
        self.failUnless(net._node_num == 2)
        del net
        print('ok.')

    def test_node_setter(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format('nodes.setter'))
        net = Network()
        int_param = 1
        with self.assertRaises(TypeError):
            net.nodes = int_param
        sys_tuple = (System(), System(), System())
        net.nodes = sys_tuple
        for i in range(3):
            self.failUnless(net.nodes[i] == sys_tuple[i])
        del net
        print('ok.')

    def test_c_setter(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format('c.setter'))
        net = Network()
        float_param = 1.0
        with self.assertRaises(TypeError):
            net.c = float_param
        with self.assertRaises(ValueError):
            net.c = np.arange(10, dtype=float)
        del net
        print('ok.')

    def test_change_link_strength(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format(Network.change_link_strength.__name__))

        initial_coupling = np.array([[-4, 1, 1, 1, 1],
                                     [1, -3, 1, 0, 1],
                                     [1, 1, -3, 1, 0],
                                     [1, 0, 1, -3, 1],
                                     [1, 1, 0, 1, -3]], dtype=float)
        net5 = Network(nodes=(System(), System(), System(), System(), System()), c=initial_coupling)
        self.failUnless(np.allclose(net5._c, initial_coupling))
        net5.change_link_strength((1, 4), 0.5)
        modified_coupling = np.array([[-4, 1, 1, 1, 1],
                                      [1, -3.5, 1, 0, 1.5],
                                      [1, 1, -3, 1, 0],
                                      [1, 0, 1, -3, 1],
                                      [1, 1.5, 0, 1, -3.5]], dtype=float)
        self.failUnless(np.allclose(net5._c, modified_coupling))
        del net5
        print('ok.')

    def test_remove_link(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format(Network.remove_link.__name__))
        initial_coupling = np.array([[-4, 1, 1, 1, 1],
                                     [1, -3, 1, 0, 1],
                                     [1, 1, -3, 1, 0],
                                     [1, 0, 1, -3, 1],
                                     [1, 1, 0, 1, -3]], dtype=float)
        net5 = Network(nodes=(System(), System(), System(), System(), System()), c=initial_coupling)
        net5.remove_link((0, 2))
        modified_coupling = np.array([[-3, 1, 0, 1, 1],
                                      [1, -3, 1, 0, 1],
                                      [0, 1, -2, 1, 0],
                                      [1, 0, 1, -3, 1],
                                      [1, 1, 0, 1, -3]], dtype=float)
        self.failUnless(np.allclose(net5._c, modified_coupling))
        del net5
        print('ok.')

    def test_add_link(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format(Network.add_link.__name__))
        initial_coupling = np.array([[-4, 1, 1, 1, 1],
                                     [1, -3, 1, 0, 1],
                                     [1, 1, -3, 1, 0],
                                     [1, 0, 1, -3, 1],
                                     [1, 1, 0, 1, -3]], dtype=float)
        net5 = Network(nodes=(System(), System(), System(), System(), System()), c=initial_coupling)
        net5.add_link((1, 3))
        modified_coupling = np.array([[-4, 1, 1, 1, 1],
                                      [1, -4, 1, 1, 1],
                                      [1, 1, -3, 1, 0],
                                      [1, 1, 1, -4, 1],
                                      [1, 1, 0, 1, -3]], dtype=float)
        self.failUnless(np.allclose(net5._c, modified_coupling))

        with self.assertRaises(IndexError):
            net5.add_link((0, 1))
        del net5
        print('ok.')

    def test_rewire_link(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format(Network.rewire_link.__name__))
        initial_coupling = np.array([[-4, 1, 1, 1, 1],
                                     [1, -3, 1, 0, 1],
                                     [1, 1, -3, 1, 0],
                                     [1, 0, 1, -3, 1],
                                     [1, 1, 0, 1, -3]], dtype=float)
        net5 = Network(nodes=(System(), System(), System(), System(), System()), c=initial_coupling)
        net5.rewire_link((1, 4), (2, 4))
        modified_coupling = np.array([[-4, 1, 1, 1, 1],
                                      [1, -2, 1, 0, 0],
                                      [1, 1, -4, 1, 1],
                                      [1, 0, 1, -3, 1],
                                      [1, 0, 1, 1, -3]], dtype=float)
        self.failUnless(np.allclose(net5._c, modified_coupling))
        del net5
        print('ok.')

    def test_compose(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format(Network.compose.__name__))
        net = Network()
        self.failUnless(hasattr(net, 'compose'))
        self.failUnless(hasattr(net, '_network_system_func'))
        self.failUnless(hasattr(net, '_network_input_func'))
        x = np.random.randn(net._node_num * net._node_dim)
        t = np.random.rand()
        f = net._network_system_func(x)
        u = net._network_input_func(x)
        xdot = net._network_rhs(x, t)
        self.failUnless(np.allclose(f + u, xdot))
        del net
        print('ok.')

    def test_set_network_system_func(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format(Network.set_network_system_func.__name__))
        net = Network()
        x = np.random.rand(net._node_num * net._node_dim)
        res1 = np.zeros(net._node_num * net._node_dim)
        for i in range(net._node_num):
            res1[i * net._node_dim: (i + 1) * net._node_dim] = net._nodes[i]._system_func(
                x[i * net._node_dim: (i + 1) * net._node_dim])

        res2 = net._network_system_func(x)
        self.failUnless(np.allclose(res1, res2))
        del net
        print('ok.')

    def test_set_network_input_func(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format(Network.set_network_input_func.__name__))
        net = Network()
        t = np.random.rand()
        res1 = net.eps * np.kron(net._c, net._p).dot(net._network_state)
        res2 = net._network_input_func(t)
        self.failUnless(np.allclose(res1, res2))
        del net
        print('ok.')

    def test_state_transition(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format(Network.state_transition.__name__))
        net = Network()
        t0 = 0.0
        t = default.t_step
        x0 = net._network_initials
        tt = np.arange(t0, t, default.t_sample)
        res1 = odeint(net._network_rhs, x0, tt)[-1]
        res2 = net.state_transition(t, t0, x0)
        self.failUnless(np.allclose(res1, res2))
        del net
        print('ok.')

    def test_write_node_buffers(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format(Network.write_node_buffers.__name__))
        net1 = Network()

        input_data = np.random.rand(net1._node_dim * net1._node_dim)
        state_data = np.random.rand(net1._node_dim * net1._node_dim)
        output_data = np.random.rand(net1._node_dim * net1._node_dim)

        net1._network_input = input_data
        net1._network_state = state_data
        net1._network_output = output_data

        net1.write_node_buffers()

        for i in range(net1._node_num):
            self.failUnless(np.allclose(net1.nodes[i]._input_buffer[0],
                                        input_data[i * net1._node_dim: (i + 1) * net1._node_dim]))
            self.failUnless(np.allclose(net1.nodes[i]._state_buffer[0],
                                        state_data[i * net1._node_dim: (i + 1) * net1._node_dim]))
            self.failUnless(np.allclose(net1.nodes[i]._output_buffer[0], net1.nodes[i]._output_func(
                state_data[i * net1._node_dim: (i + 1) * net1._node_dim], 0.0)))
        print('ok.')

    def test_call(self):
        print('\nTest: {}.'.format(Network.__name__) + '{}'.format(Network.__call__.__name__))
        net1 = Network()

        delta_t = default.t_step
        x0 = net1._network_initials
        t0 = 0.0
        tt = np.arange(t0, delta_t, default.t_sample)
        # self.failUnless(np.allclose(net1._network_input, zero_func(t0)))
        next_state = odeint(net1._network_rhs, x0, tt)[-1]

        net1()  # run the system for delta_t = default.t_step

        self.failUnless(np.allclose(net1._network_state, next_state))
        print('ok')
