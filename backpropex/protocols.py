"""
Protocols to allow the various classes to collaborate.
"""

from typing import Sequence, Optional, overload, runtime_checkable, Protocol, TYPE_CHECKING, Any, Generator
from networkx import DiGraph

from backpropex.types import (
    FloatSeq, NPFloats, TrainingData,
)
from backpropex.steps import (
    EvalStepResultAny,
    InitStepResult,
    TrainStepResultAny
)

if TYPE_CHECKING:
    from backpropex.layer import Layer
    from backpropex.edge import Edge
    from backpropex.node import Node
    from backpropex.loss import LossFunction


@runtime_checkable
class EvalProtocol(Protocol):
    """
    A neural network that can be evaluated.

    This is the public protocol to evaluate a network on a given input.
    """
    def __call__(self, input: FloatSeq, /, *,
                label: Optional[str] = None
                ) -> Generator[EvalStepResultAny|InitStepResult, Any, None]:
        ...
    net: 'NetProtocol'

@runtime_checkable
class NetProtocol(EvalProtocol, Protocol):
    """
    A neural network. This is the public protocol by which other classes interact with the network.
    """
    net: 'NetProtocol'
    graph: DiGraph = DiGraph()
    layers: Sequence['Layer']
    max_layer_size: int
    name: str

    # Progress information for drawing the network
    # The layer that is currently being evaluated
    active_layer: Optional['Layer'] = None
    active_message: Optional[str] = None

    @property
    def labels(self) -> dict['Node', str]:
        ...

    @property
    def weights(self) -> Generator[float, None, None]:
        ...

    @property
    def values(self) -> Generator[float, None, None]:
        ...

    @property
    def edges(self) -> Generator['Edge', None, None]:
        ...

    @property
    def nodes(self) -> Generator['Node', None, None]:
        ... # type: ignore

    @property
    def real_nodes(self) -> Generator['Node', None, None]:
        ...

    def in_edges(self, node: 'Node') -> Generator['Edge', None, None]:
        ...

    @property
    def input_layer(self) -> 'Layer':
        ...

    @property
    def output_layer(self) -> 'Layer':
        ...

    @property
    def input(self) -> tuple[float, ...]:
        ...

    @property
    def output(self) -> tuple[float, ...]:
        ...

    @property
    def output_array(self) -> NPFloats:
        ...
    @property
    def hidden_layers(self) -> tuple['Layer', ...]:
        ...

    def __getitem__(self, idx: int) -> 'Layer':
        ...

    def __len__(self) -> int:
        ...

@runtime_checkable
class TrainProtocol(Protocol):
    """
    A trainer for neural networks.
    This composes with the network to train it.
    """
    def __init__(self, network: NetProtocol, /, *, loss_function: 'LossFunction'):
        ...
    def __call__(self, data: TrainingData, /, *,
              epochs: int=1000,
              learning_rate: float=0.1
              ) -> Generator[TrainStepResultAny|InitStepResult, Any, None]:
        ...
    net: NetProtocol

@runtime_checkable
class GraphProtocol(Protocol):
    """
    A wrapper for a network to draw it as a graph. This composes with the network,
    either with or without a Trainer.
    """
    @overload
    def __init__(self, net: NetProtocol, /, *,
                 margin: float=0.13,
                 ) -> None:
        ...
    @overload
    def __init__(self, trainer: TrainProtocol, /, *,
                 margin: float=0.13,
                 ) -> None:
        ...

    def __init__(self, proxy: NetProtocol|TrainProtocol, /, *,
                 margin: float=0.13,
                 ) -> None:
        ...

class LossFunction(Protocol):
    """
    The protocol for an Loss function.
    """

    name: str
    def __call__(self, actual: NPFloats, expected: NPFloats, /) -> float:
        ...

    def derivative(self, actual: NPFloats, expected: NPFloats, /) -> NPFloats:
        ...

class ActivationFunction(Protocol):
    """
    The protocol for an activation function.
    """

    name: str
    def __call__(self, x: float) -> float:
        ...

    def derivative(self, x: float) -> float:
        ...

__all__ = ['EvalProtocol', 'NetProtocol', 'TrainProtocol', 'GraphProtocol', 'LossFunction', 'ActivationFunction']
