"""
Randomizer module for backpropex. A randomizer is a function that returns a random matrix of the given shape.
matrix of random numbers of the given shape, used to initialize the weights
connecting two layers in the network.
"""
import numpy as np

from backpropex.types import NPFloat2D
from backpropex.protocols import Randomizer

class HeEtAl(Randomizer):
    """
    A randomizer that generates random numbers according to the He-et-al initialization.
    """
    def __call__(self, shape: tuple[int, int], /) -> NPFloat2D:
        """
        Return a matrix of random numbers of the given shape, used to initialize the weights
        connecting two layers in the network.
        """
        # He-et-al initialization
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])
