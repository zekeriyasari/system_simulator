from systemssimulator.systems import *
import numbers
import types


def check_parameter_type(param, expected_type):
    flag = True
    if isinstance(expected_type, tuple):
        for type_ in expected_type:
            flag &= isinstance(param, type_)

    flag &= isinstance(param, expected_type)

    if not flag:
        raise TypeError('Expected {}, got {}'.format(expected_type.__name__, type(param)))

    return flag


class Network(object):
    def __init__(self,
                 nodes=(System(), System()),
                 a=np.array([[0., -1.], [-1., 0]]),
                 p=np.diag([1., 0., 0.]),
                 eps=default.epsh):
        self._nodes = None
        self._a = None
        self._p = None
        self._eps = None

        self.nodes = nodes
        self._node_dim = self.nodes[0]._dim
        self._node_num = len(self.nodes)
        self._c = np.zeros((self._node_num, self._node_num))
        self.a = a
        self.p = p
        self.eps = eps

        self._network_system_func = self.set_network_system_func()
        self._network_input_func = self.set_network_input_func()
        self._network_rhs = self.compose(self._network_system_func, self._network_input_func)

        self._network_state = np.hstack((self.nodes[i]._state for i in range(self._node_num)))
        self._network_input = np.hstack((self.nodes[i]._input for i in range(self._node_num)))
        self._network_output = np.hstack((self.nodes[i]._output for i in range(self._node_num)))

        self._network_initials = np.hstack((self._nodes[i]._initials for i in range(self._node_num)))
        self._network_t = self._nodes[0]._t
        self._network_t_clock = self._nodes[0]._t_clock
        self._buffer_index = 0

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        check_parameter_type(nodes, tuple)
        if not len(nodes) >= 2:
            raise ValueError('Number of nodes in a network must be greater than 1')
        for node in nodes:
            check_parameter_type(node, System)
            if not getattr(node, '_dim') == nodes[0]._dim:
                raise ValueError('Node dimensions must be equal')
        self._nodes = nodes

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, val):
        check_parameter_type(val, np.ndarray)
        assert val.dtype == 'float', 'Expected {} entries, got {} entries'.format(float.__name__, val.dtype)
        if not val.shape == (self._node_num, self._node_num):
            raise ValueError('Expected shape {}, got shape{}'.format((self._node_num, self._node_num), val.shape))
        if not np.allclose(val.diagonal(), np.zeros(self._node_num)):
            raise ValueError('Matrix must have all-zero diagonal')
        if not (val <= 0.).all():
            raise ValueError('Matrix must have non-positive entries')
        self._a = val
        self.coupling_from_adjacency()

    def coupling_from_adjacency(self):
        """Create diagonalizes coupling matrix from adjacency matrix"""
        self._c = np.copy(self._a)
        if (self._c == 0.).all():
            return
        for i in range(self._a.shape[0]):
            self._c[i, i] = -sum(self._a[i])
            # self._c[i] /= self._c[i, i]

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, val):
        check_parameter_type(val, np.ndarray)
        assert val.dtype == 'float', 'Expected {} entries, got {}'.format(float.__name__, val.dtype)
        if not val.shape == (self._node_dim, self._node_dim):
            raise ValueError('Expected shape {}, got shape{}'.format((self._node_dim, self._node_dim), val.shape))
        self._p = val

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, val):
        check_parameter_type(val, float)
        if not val >= 0.0:
            raise ValueError('Coupling strength must be positive')
        self._eps = val

    def change_link_strength(self, link, weight):
        check_parameter_type(weight, numbers.Number)
        if not weight <= 0.0:
            raise ValueError('Weight must be non-positive')
        check_parameter_type(link, tuple)

        i, j = link
        # update the adjacency matrix
        self._a[i, j] = weight
        self._a[j, i] = weight

        # update the coupling matrix
        self.coupling_from_adjacency()

    def remove_link(self, link):
        check_parameter_type(link, tuple)
        i, j = link
        weight = self._a[i, j]

        # update the adjacency matrix
        self._a[i, j] = 0.
        self._a[j, i] = 0.

        # update the coupling matrix
        self.coupling_from_adjacency()

    def add_link(self, link, weight=-1.0):
        check_parameter_type(link, tuple)
        i, j = link

        if not (self._c[i, j] == 0 and self._c[j, i] == 0):
            raise IndexError('Link {} already exist'.format(link))

        self.change_link_strength(link, weight=weight)

    def rewire_link(self, link_remove, link_add):
        check_parameter_type(link_remove, tuple)
        check_parameter_type(link_add, tuple)

        self.remove_link(link_remove)
        self.add_link(link_add)

    @staticmethod
    def compose(system_function, input_function):
        check_parameter_type(system_function, types.FunctionType)
        check_parameter_type(input_function, types.FunctionType)

        def composed(x, t):
            return system_function(x) + input_function(t)

        return composed

    def set_network_system_func(self):
        def system_func(x):
            res = np.zeros(self._node_num * self._node_dim)
            for i in range(self._node_num):
                res[i * self._node_dim: (i + 1) * self._node_dim] = self._nodes[i]._system_func(
                    x[i * self._node_dim: (i + 1) * self._node_dim])
            return res

        return system_func

    def set_network_input_func(self):
        def input_func(t):
            return -self.eps * np.kron(self._c, self._p).dot(self._network_state)

        return input_func

    def state_transition(self, t, t0, x0):
        tt = np.arange(t0, t, default.t_sample)
        res = odeint(self._network_rhs, x0, tt)[-1]  # numerically integrate the system
        return res

    def write_node_buffers(self):
        for i in range(self._node_num):
            node = self._nodes[i]
            node_input = self._network_input[i * self._node_dim: (i + 1) * self._node_dim]
            node_state = self._network_state[i * self._node_dim: (i + 1) * self._node_dim]
            node_output = node._output_func(node_state, self._network_t)

            node._input_buffer[node._buffer_index] = node_input
            node._state_buffer[node._buffer_index] = node_state
            node._output_buffer[node._buffer_index] = node_output
            node._buffer_index += 1

    def __call__(self, delta_t=default.t_step):
        # read the coupling input at time t.
        self._network_input = self._network_input_func(self._network_t)

        # calculate the next state at time t + delta_t
        self._network_state = self.state_transition(self._network_t + delta_t, self._network_t, self._network_state)

        # write node buffers
        self.write_node_buffers()

        # update the network time.
        self._network_t += default.t_sample

    def run(self, t_run=default.t_max, t_step=default.t_step, initial_time=0.0):
        self._network_t = initial_time
        while self._network_t_clock <= t_run:
            self(delta_t=t_step)
            self._network_t_clock += t_step

    def __str__(self):
        return '{} at {:x}\n' \
               'c: {}\n' \
               'p: {}\n' \
               'eps {}\n'.format(self.__class__.__qualname__, id(self),
                                 self._c, self._p, self.eps)
