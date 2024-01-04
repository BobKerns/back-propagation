"""
This module contains type definitions for the backpropex package.

Separating the type definitions from the code allows the code to be type-checked
without risking circular imports.
"""

from enum import StrEnum
import numpy as np
from numpy.typing import NDArray

type NPArray = NDArray[np.float_]

class LayerType(StrEnum):
    """
    The type of a layer in a neural network.
    """
    Input = "Input"
    Hidden = "Hidden"
    Output = "Output"
