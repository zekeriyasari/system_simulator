import numpy as np


class Dummy(object):
    def __init__(self, inits=None):
        if inits is not None:
            self._initials = inits
        else:
            self._initials = np.random.randn(4)

obj = Dummy()
print(obj._initials)
obj = Dummy()
print(obj._initials)
