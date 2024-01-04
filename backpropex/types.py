"""
This module contains type definitions for the backpropex package.

Separating the type definitions from the code allows the code to be type-checked
without risking circular imports.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal, Optional, TYPE_CHECKING, Protocol, TypedDict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from backpropex.layer import Layer

type NPFloats = NDArray[np.float_|np.int_]
type FloatSeq = Sequence[float|int]|tuple[float|int]|NPFloats

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

class StepType(StrEnum):
    """
    The type of a step in a training session.
    """
    Initialized = "Initialized"
    Input = "Input"
    Forward = "Forward"
    Output = "Output"
    TrainInput = "TrainInput"
    TrainForward = "TrainForward"
    TrainOutput = "TrainOutput"
    TrainLoss = "TrainLoss"
    TrainBackward = "TrainBackward"
    TrainOptimize = "Optimize"

# Step types for ordinary evaluation
type EvalStepType = Literal[StepType.Input, StepType.Forward, StepType.Output]

# Step types for training
type TrainStepType = Literal[
    StepType.TrainInput,
    StepType.TrainForward,
    StepType.TrainOutput,
    StepType.TrainLoss,
    StepType.TrainBackward,
    StepType.TrainOptimize
]

type LayerStepType = EvalStepType|TrainStepType

# Step types that provide input
type InputStepType = Literal[StepType.Input, StepType.TrainInput]

# Step types that produce output
type OutputStepType = Literal[StepType.Output, StepType.TrainOutput]

@dataclass
class StepResult[T: StepType]:
    """
    Results from a step in processing a network.

    :param type: The type of step.
    """
    type: T


@dataclass
class LayerStepResult[T: LayerStepType](StepResult[T]):
    """
    Results from a step in processing a layer a network.

    :param type: The type of step. (inherited)
    :param layer: The layer that was processed.
    """
    layer: 'Layer'

@dataclass
class EvalStepResult[T: EvalStepType](LayerStepResult[T]):
    pass

@dataclass
class InputStepResult[T: InputStepType](LayerStepResult[T]):
    """
    Results from a step in processing a layer a network.

    :param type: The type of step. (inherited)
    """
    input: tuple[float, ...]

@dataclass
class OutputStepResult[T: OutputStepType](LayerStepResult[T]):
    output: tuple[float, ...]

@dataclass
class TrainStepResult[T: TrainStepType](LayerStepResult[T]):
    epoch: int
    epoch_max: int
    datum_no: int
    datum_max: int
    datum: TrainingDatum

# Concrete step results
@dataclass
class InitStepResult(StepResult[Literal[StepType.Initialized]]):
    pass

@dataclass
class EvalInputStepResult(EvalStepResult[Literal[StepType.Input]],
                          InputStepResult[Literal[StepType.Input]]):
    pass

@dataclass
class EvalForwardStepResult(EvalStepResult[Literal[StepType.Forward]],
                            LayerStepResult[Literal[StepType.Forward]]):
    pass

@dataclass
class EvalOutputStepResult(EvalStepResult[Literal[StepType.Output]],
                          OutputStepResult[Literal[StepType.Output]]):
    pass

type EvalStepResultAny = EvalInputStepResult|EvalForwardStepResult|EvalOutputStepResult
@dataclass
class TrainInputStepResult(InputStepResult[Literal[StepType.TrainInput]],
                           TrainStepResult[Literal[StepType.TrainInput]]):
    pass
@dataclass
class TrainForwardStepResult(TrainStepResult[Literal[StepType.TrainForward]],
                             LayerStepResult[Literal[StepType.TrainForward]]):
    pass

@dataclass
class TrainOutputStepResult(TrainStepResult[Literal[StepType.TrainOutput]],
                            OutputStepResult[Literal[StepType.TrainOutput]]):
    pass

@dataclass
class TrainLossStepResult(TrainStepResult[Literal[StepType.TrainLoss]]):
    loss: Optional[float]

@dataclass
class TrainBackwardStepResult(TrainStepResult[Literal[StepType.TrainBackward]]):
    gradient: NPFloats

class TrainOptimizeStepResult(TrainStepResult[Literal[StepType.TrainOptimize]]):
    weight_delta: NPFloats
    weight: NPFloats

type TrainStepResultAny = TrainInputStepResult|TrainForwardStepResult|TrainOutputStepResult\
    |TrainLossStepResult|TrainBackwardStepResult|TrainOptimizeStepResult

__all__ = [
    'NPFloats', 'FloatSeq', 'TrainingDatum', 'TrainingData',
    'LayerType',
    'NetTuple', 'TrainingInfo',
    'StepType', 'TrainStepType', 'OutputStepType', 'LayerStepType',
    'StepResult', 'InitStepResult', 'LayerStepResult', 'OutputStepResult', 'TrainStepResult',
    'EvalInputStepResult', 'EvalForwardStepResult', 'EvalOutputStepResult', 'EvalStepResultAny',
    'TrainInputStepResult', 'TrainForwardStepResult', 'TrainOutputStepResult',
    'TrainLossStepResult', 'TrainBackwardStepResult', 'TrainOptimizeStepResult', 'TrainStepResultAny'
    ]
