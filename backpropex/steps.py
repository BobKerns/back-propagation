"""
The various steps that can be taken in the network and their result data.
"""


from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Optional, TYPE_CHECKING, Sequence

from backpropex.types import NPFloats, TrainingDatum

if TYPE_CHECKING:
    from backpropex.layer import Layer


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
type EvalStepType = Literal[StepType.Input, StepType.Forward, StepType.Output, StepType.Initialized]

# Step types for training
type TrainStepType = Literal[
    StepType.TrainInput,
    StepType.TrainForward,
    StepType.TrainOutput,
    StepType.TrainLoss,
    StepType.TrainBackward,
    StepType.TrainOptimize,
    StepType.Initialized
]

type LayerStepType = EvalStepType|TrainStepType

# Step types that provide input
type InputStepType = Literal[StepType.Input, StepType.TrainInput]

# Step types that produce output
type OutputStepType = Literal[StepType.Output, StepType.TrainOutput]

type StepTypeAny = StepType|EvalStepType|TrainStepType|Literal[StepType.Initialized]

@dataclass
class StepResult[T: (StepTypeAny,StepType)]:
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
    values: Sequence[float]

@dataclass
class EvalOutputStepResult(EvalStepResult[Literal[StepType.Output]],
                          OutputStepResult[Literal[StepType.Output]]):
    pass

type EvalStepResultAny = EvalInputStepResult|EvalForwardStepResult\
    |EvalOutputStepResult|InitStepResult
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
    |TrainLossStepResult|TrainBackwardStepResult|TrainOptimizeStepResult\
    |InitStepResult

type StepResultAny = EvalStepResultAny|TrainStepResultAny

__all__ = [
    'StepType', 'TrainStepType', 'OutputStepType', 'LayerStepType', 'StepTypeAny',
    'StepResult', 'InitStepResult', 'LayerStepResult', 'OutputStepResult', 'TrainStepResult',
    'EvalInputStepResult', 'EvalForwardStepResult', 'EvalOutputStepResult', 'EvalStepResultAny',
    'TrainInputStepResult', 'TrainForwardStepResult', 'TrainOutputStepResult',
    'TrainLossStepResult', 'TrainBackwardStepResult', 'TrainOptimizeStepResult', 'TrainStepResultAny',
    'StepResultAny'
    ]
