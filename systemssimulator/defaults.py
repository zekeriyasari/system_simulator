

DISCRETE_TIME = 0
CONTINUOUS_TIME = 1

SystemType = {0: 'Discrete Time System', 1: 'Continuous Time System'}


class Defaults(object):
    buffer_size = 100000
    t_max = 100.0
    t_sample = 0.001
    t_step = 0.01
    dim = 3

default = Defaults()

