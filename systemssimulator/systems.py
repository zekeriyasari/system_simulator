from systemssimulator.defaults import *
import numpy as np
from scipy.integrate import odeint
from inspect import signature


def lorenz(x, sigma=10.0, b=8.0 / 3.0, r=35.0):
    xdot = sigma * (x[1] - x[0])
    ydot = x[0] * (r - x[2]) - x[1]
    zdot = x[0] * x[1] - b * x[2]
    return np.array([xdot, ydot, zdot])


def zero_func(t):
    return np.zeros(default.dim)


def get_x_state(x, t, c=np.diag([1, 0, 0])):
    return c.dot(x)


class System(object):
    def __init__(self,
                 system_type=CONTINUOUS_TIME,
                 dim=default.dim,
                 system_func=lorenz,
                 input_func=zero_func,
                 output_func=get_x_state,
                 initials=None,  # if np.array is set as default, then all system objects have same initial conditions.
                 buffer_size=default.buffer_size):
        self._dim = dim
        self._type = system_type

        self._system_func = system_func
        self._input_func = input_func
        self._output_func = output_func

        self._input_buffer = np.zeros((buffer_size, self._dim))
        self._state_buffer = np.zeros((buffer_size, self._dim))
        self._output_buffer = np.zeros((buffer_size, self._dim))

        self._rhs = self.compose(self._system_func, self._input_func)
        self._initials = initials if initials is not None else np.random.randn(self._dim)
        self._input = np.zeros(self._dim)
        self._output = np.zeros(self._dim)
        self._state = self._initials
        self._t = 0.0
        self._t_clock = 0.0
        self._buffer_index = 0

        self.input_func = input_func

    @staticmethod
    def compose(system_function, input_function):
        def composed(x, t):
            return system_function(x) + input_function(t)

        return composed

    @property
    def input_func(self):
        return self._input_func

    @input_func.setter
    def input_func(self, val):
        assert callable(val), 'Expected {}'.format(callable.__name__) + 'but got {}'.format(type(val))

        sig = signature(val)
        for param in sig.parameters.values():
            if not param.kind == param.POSITIONAL_OR_KEYWORD:
                raise TypeError('Expected positional parameters, got {}'.format(param.kind))

        self._input_func = val

    def __str__(self):
        return '{} at           {:x}\n' \
               'system_function {}:\n' \
               'input function  {}:\n' \
               'output function {}:\n' \
               'system time     {}:\n'.format(self.__class__.__qualname__, id(self), self._system_func.__name__,
                                              self._input_func.__name__, self._output_func.__name__, self._t_clock)

    def __call__(self, delta_t=default.t_step):
        # read the input at time t.
        self._input = self._input_func(self._t)

        # calculate the next state at time t + delta_t
        self._state = self.state_transition(self._t + delta_t, self._t, self._state)

        # calculate the output at time t + _delta_t
        self._output = self._output_func(self._state, self._t)

        # write the buffers
        self.write_buffers()

        # update system time.
        self._t += default.t_sample

    def state_transition(self, t, t0, x0):
        tt = np.arange(t0, t, default.t_sample)
        res = odeint(self._rhs, x0, tt)[-1]  # numerically integrate the system
        return res

    def write_buffers(self):
        self._input_buffer[self._buffer_index] = self._input
        self._state_buffer[self._buffer_index] = self._state
        self._output_buffer[self._buffer_index] = self._output
        self._buffer_index += 1

    def read_buffer(self, buffer_name):
        return getattr(self, buffer_name)[:self._buffer_index]

    def run(self, t_run=default.t_max, t_step=default.t_step, initial_time=0.0):
        self._t = initial_time
        while self._t_clock <= t_run:
            self(delta_t=t_step)
            self._t_clock += default.t_step
