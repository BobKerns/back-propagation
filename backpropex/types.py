"""
This module contains type definitions for the backpropex package.

Separating the type definitions from the code allows the code to be type-checked
without risking circular imports.
"""

from collections.abc import Sequence
from enum import StrEnum
import numpy as np
from numpy.typing import NDArray

type NPFloats = NDArray[np.float_]
type FloatSeq = Sequence[float]|NPFloats

class LayerType(StrEnum):
    """
    The type of a layer in a neural network.
    """
    Input = "Input"
    Hidden = "Hidden"
    Output = "Output"
