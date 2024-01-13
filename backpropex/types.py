"""
This module contains type definitions for the backpropex package.

Separating the type definitions from the code allows the code to be type-checked
without risking circular imports.
"""

from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import NamedTuple, Protocol, TypedDict

import numpy as np

# include np.int_ in NPFloats to allow for integer inputs
type NPFloat1D = np.ndarray[int, np.dtype[np.float_]]
type FloatSeq = Sequence[float|int]|tuple[float|int]|NPFloat1D
type NPFloat2D = np.ndarray[tuple[int, int], np.dtype[np.float_]]
type NPObject1D = np.ndarray[int, np.dtype[np.object_]]
type NPObject2D = np.ndarray[tuple[int, int], np.dtype[np.object_]]

type RGBA = tuple[float, float, float, float]

type Point = tuple[float, float]
class NetTuple(Protocol):
    """Constructor for input or output named tuples"""
    def __call__(self, *args: float) -> tuple[float, ...]:
        ...


class TrainingItem(NamedTuple):
    """
    A training item is a tuple of input and expected output, to which we add
    an ID to track it through the shuffling of the training process.
    """
    input: NPFloat1D
    expected: NPFloat1D
    id: int

type TrainingSet=list[TrainingItem]

type TrainingDatum = tuple[FloatSeq, FloatSeq]
type TrainingData = Iterable[TrainingDatum]

class TrainingProgress(TypedDict):
    """Information about the ongoing training process. """
    epoch: int
    epoch_max: int
    datum_no: int
    datum_max: int
    datum: TrainingItem

class LayerType(StrEnum):
    """
    The type of a layer in a neural network.
    """
    Input = "Input"
    Hidden = "Hidden"
    Output = "Output"

__all__ = [
    'NPFloat1D', 'FloatSeq', 'NPFloat2D', 'RGBA', 'Point',
    'NPObject1D', 'NPObject2D',
    'TrainingDatum', 'TrainingData',
    'LayerType',
    'NetTuple', 'TrainingProgress', 'TrainingItem', 'TrainingSet',
]
