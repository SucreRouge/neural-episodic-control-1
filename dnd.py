import numpy as np


class DND(object):
    """Numpy implementation of Differentiable Neural Dictionary"""

    def __init__(self, num_act, num_mem):
        self.num_act = num_act
        self.num_mem = num_mem

        self.key = np.empty((num_act, num_mem, 512))
        self.value = np.empty((num_act, num_mem, 1))