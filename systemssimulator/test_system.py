from unittest import TestCase
from systemssimulator.systems import *
import matplotlib.pyplot as plt


class TestSystem(TestCase):
    sys = None

    def setUp(self):
        self.sys = System()

    def tearDown(self):
        del self.sys

    def test_system_initialize(self):
        print('\nTest: {}.'.format(System.__name__) + '{}'.format(System.__call__.__name__))
        self.sys = System()
        self.failUnless(isinstance(self.sys, System))
        self.failUnless(self.sys._dim == default.dim)
        self.failUnless(self.sys._type == CONTINUOUS_TIME)
        self.failUnless(self.sys._t == 0.0)
        self.failUnless(np.allclose(self.sys._initials, np.zeros(default.dim)))
        self.failUnless(self.sys._input_func == zero_func)
        self.failUnless(self.sys._system_func == lorenz)
        self.failUnless(self.sys._output_func == get_x_state)
        self.failUnless(np.allclose(self.sys._input_buffer, np.zeros((default.buffer_size, self.sys._dim))))
        self.failUnless(np.allclose(self.sys._state_buffer, np.zeros((default.buffer_size, self.sys._dim))))
        self.failUnless(np.allclose(self.sys._output_buffer, np.zeros((default.buffer_size, self.sys._dim))))
        print('ok...')

    def test_compose(self):
        print('\nTest: {}.'.format(System.__name__) + '{}'.format(System.compose.__name__))
        self.failUnless(hasattr(System, 'compose') and callable(getattr(System, 'compose')))

        def f(x):
            return 2 * x

        def u(x, t):
            return x + t

        rhs = self.sys.compose(f, u)
        x = np.random.randn(self.sys._dim)
        t = np.random.randn()
        self.failUnless(np.allclose(f(x) + u(x, t), rhs(x, t)))
        print('ok.')

    def test_input_func_setter(self):
        print('\nTest: {}.'.format(System.__name__) + '{}'.format('input_func.setter'))

        a = 1

        def f(x):
            return x

        def g(x, t, *args, **kwargs):
            return

        def h(*args):
            return

        with self.assertRaises(AssertionError):
            self.sys.input_func = a

        with self.assertRaises(TypeError):
            self.sys.input_func = h

        with self.assertRaises(TypeError):
            self.sys.input_func = g

    def test_str(self):
        print('\nTest: {}.'.format(System.__name__) + '{}'.format(System.__str__.__name__))
        print(self.sys)
        print('ok')

    def test_rhs(self):
        print('\nTest: {}.'.format(System.__name__) + '{}'.format(self.sys._rhs.__name__))
        self.failUnless(callable(self.sys._rhs))
        self.failUnless(callable(self.sys._system_func))
        self.failUnless(callable(self.sys._input_func))
        x = np.random.randn(self.sys._dim)
        t = np.random.rand()
        self.failUnless(np.allclose(self.sys._system_func(x) + self.sys._input_func(x, t), self.sys._rhs(x, t)))
        print('ok.')

    def test_call(self):
        print('\nTest: {}'.format(System.__call__.__name__))

        delta_t = default.t_step
        x0 = self.sys._initials
        t0 = self.sys._t
        tt = np.arange(t0, delta_t, default.t_sample)
        self.failUnless(np.allclose(self.sys._input, zero_func(x0, t0)))
        next_state = odeint(self.sys._rhs, x0, tt)

        self.sys()  # run the system for delta_t = default.t_step

        self.failUnless(np.allclose(self.sys._state, next_state))
        print('ok')

    def test_state_transition(self):
        print('\nTest: {}.'.format(System.__name__) + '{}'.format(self.sys.state_transition.__name__))
        delta_t = default.t_step
        x0 = self.sys._initials
        t = self.sys._t + delta_t
        next_state = self.sys.state_transition(t, 0.0, x0)
        self.failUnless(next_state.shape == self.sys._state.shape)
        print('ok.')

    def test_write_buffers(self):
        print('\nTest: {}.'.format(System.__name__) + '{}'.format(self.sys.write_buffers.__name__))
        self.failUnless(self.sys._buffer_index == 0)
        for i in range(10):
            self.sys._input = np.ones(self.sys._dim)
            self.sys._state = np.arange(self.sys._dim)
            self.sys._output = np.ones(self.sys._dim)
            self.sys.write_buffers()
            self.failUnless(self.sys._buffer_index == i + 1)
            self.failUnless(np.allclose(self.sys._input_buffer[i], self.sys._input))
            self.failUnless(np.allclose(self.sys._state_buffer[i], self.sys._state))
            self.failUnless(np.allclose(self.sys._output_buffer[i], self.sys._output))
        print('ok')

    def test_read_buffer(self):
        print('\nTest: {}.'.format(System.__name__) + '{}'.format(self.sys.read_buffer.__name__))
        buf = np.zeros((10, self.sys._dim))
        for i in range(10):
            self.sys()
            buf[i] = self.sys._state
        self.failUnless(self.sys._buffer_index == 10)
        self.failUnless(np.allclose(self.sys.read_buffer('_state_buffer'), buf))
        print('ok')

    def test_run(self):
        print('\nTest: {}.'.format(System.__name__) + '{}'.format(self.sys.run.__name__))
        # sys1 = System(initials=np.array([1, 2, 3]))
        # sys2 = System(initials=np.array([1, 2, 3]))
        # x0 = np.array([1, 2, 3])
        # t0 = 0.0
        # t = default.t_max
        # tt = np.arange(t0, t, default.t_step)
        # res1 = odeint(sys1._rhs, x0, tt)[:, 0]
        # res2 = odeint(sys2._rhs, x0, tt)[:, 0]
        # sys1.run()
        # sys2.run()
        # res3 = sys1.read_buffer('_state_buffer')[:, 0]
        # res4 = sys2.read_buffer('_state_buffer')[:, 0]
        # plt.plot(res1)
        # plt.plot(res2)
        # plt.figure()
        # plt.plot(res3)
        # plt.plot(res4)
        # plt.figure()
        # plt.plot(np.abs(res1 - res3))
        # plt.show()
        pass
