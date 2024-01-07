"""
This module contains type definitions for the backpropex package.

Separating the type definitions from the code allows the code to be type-checked
without risking circular imports.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

# include np.int_ in NPFloats to allow for integer inputs
type NPFloats = NDArray[np.float_|np.int_]
type FloatSeq = Sequence[float|int]|tuple[float|int]|NPFloats
type NPFloat2D = np.ndarray[tuple[int, int], np.dtype[np.float_]]

type RGBA = tuple[float, float, float, float]

type Point = tuple[float, float]
class NetTuple(Protocol):
    """Constructor for input or output named tuples"""
    def __call__(self, *args: float) -> tuple[float, ...]:
        ...

type TrainingDatum = tuple[FloatSeq, FloatSeq]
type TrainingData = Sequence[TrainingDatum]

class TrainingInfo(TypedDict):
    epoch: int
    epoch_max: int
    datum_no: int
    datum_max: int
    datum: tuple[NPFloats, NPFloats]
class LayerType(StrEnum):
    """
    The type of a layer in a neural network.
    """
    Input = "Input"
    Hidden = "Hidden"
    Output = "Output"

__all__ = [
    'NPFloats', 'FloatSeq', 'NPFloat2D', 'RGBA', 'Point',
    'TrainingDatum', 'TrainingData',
    'LayerType',
    'NetTuple', 'TrainingInfo',
]
