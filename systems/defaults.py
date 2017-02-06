from systems.abstract import Structure

DISCRETE_SYSTEM = 0
CONTINUOUS_SYSTEM = 1

SystemType = {0: 'Discrete',
              1: 'Continuous'}


class Defaults(Structure):
    _parameters = dict(buffer_size=(64, int , "Default buffer size"),
                       t_max=(10., float, "Maximum simulation time"),
                       t_sample=(1. / 64., float, "Sampling period"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

defaults = Defaults()

