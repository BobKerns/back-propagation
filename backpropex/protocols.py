"""
Protocols to allow the various classes to collaborate.

This avoids circular imports, as we don't need to import the classes themselves,
and the protocols serve as templates for extension.

The protocols are checked at runtime, so they can be used to check the types
of objects created at runtime.

A few key types are not defined as protocols, and are imported via the
`if TYPE_CHECKING` mechanism, both here and elsewhere:

* Node
* Edge
* Layer
* LossFunction
"""

from collections.abc import Iterable
from contextlib import contextmanager
from typing import (
    Callable, Literal, Optional, overload,
    runtime_checkable, Protocol, TYPE_CHECKING,
    Any, Generator
)

from backpropex.types import (
    FloatSeq, NPFloat2D, NPFloat1D, NPObject2D, TrainingData, TrainingProgress, TrainingItem
)
from backpropex.steps import (
    EvalStepResultAny,
    FilterArg,
    InitStepResult,
    StepResult,
    StepResultAny,
    StepType,
    StepTypeAny,
    TrainBackwardStepResult,
    TrainOptimizeStepResult,
    TrainStepResultAny
)

if TYPE_CHECKING:
    from backpropex.node import Node
    from backpropex.edge import Edge
    from backpropex.layer import Layer
    from backpropex.loss import LossFunction


@runtime_checkable
class EvalProtocol(Protocol):
    """
    A neural network that can be evaluated.

    This is the public protocol to evaluate a network on a given input.
    """
    def __call__(self, input: FloatSeq, /, *,
                label: Optional[str] = None,
                **kwargs: Any
                ) -> Generator[EvalStepResultAny|InitStepResult, Any, None]:
        ...
    net: 'NetProtocol'

@runtime_checkable
class NetProtocol(EvalProtocol, Protocol):
    """
    A neural network. This is the public protocol by which other classes interact with the network.
    """
    net: 'NetProtocol'
    layers: list['Layer']
    edges_: NPObject2D
    max_layer_size: int
    name: str

    # Progress information for drawing the network
    # The layer that is currently being evaluated
    active_layer: Optional['Layer'] = None
    active_message: Optional[str] = None

    loss_function: 'LossFunction'

    @contextmanager
    def filter(self, _filter: 'Filter|type[Filter]|None', /) -> Generator['Filter|None', Any, None]:
        ...

    @contextmanager
    def step_active(self, layer: 'Layer', /) -> Generator['Layer', Any, None]:
        ...

    def filterCheck[R: StepResultAny](
        self, type: StepType,
        mk_step: Callable[[], R],
        /,) -> Generator[R, Any, None]:
        ...

    @contextmanager
    def trace(self, trace: 'Trace|type[Trace]|None', /) -> Generator['Trace|None', Any, None]:
        ...

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
    def output_array(self) -> NPFloat1D:
        ...
    @property
    def hidden_layers(self) -> tuple['Layer', ...]:
        ...

    def __getitem__(self, idx: int) -> 'Layer':
        ...

    def __len__(self) -> int:
        ...

@runtime_checkable
class LossFunction(Protocol):
    """
    The protocol for an Loss function.
    """

    name: str
    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        ...

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                   **kwargs: Any) -> NPFloat1D:
        ...

@runtime_checkable
class TrainProtocol(Protocol):
    """
    A trainer for neural networks.
    This composes with the network to train it.
    """
    def __init__(self, network: NetProtocol, /, *,
                 loss_function: LossFunction):
        ...
    def __call__(self, data: TrainingData, /, *,
              epochs: int=1000,
              learning_rate: float=0.1,
              **kwargs: Any
              ) -> Generator[TrainStepResultAny|InitStepResult, Any, None]:
        ...
    net: NetProtocol
    loss_function: LossFunction


@runtime_checkable
class BackpropagateProtocol(Protocol):
    """
    A module to backpropagate the error gradient through the network.
    """
    def __call__(self, trainer: TrainProtocol, training_item: TrainingItem, /, *,
                 training_progress: TrainingProgress,
                 **kwargs: Any) -> Generator[TrainBackwardStepResult, Any, None]:
        """
        Backpropagate the gradient through the network.
        """
        ...

@runtime_checkable
class OptimizerProtocol(Protocol):
    """
    A protocol for an optimizer.
    """
    def __call__(self, net: NetProtocol, loss: float, /, *,
                 epochs: int=1000,
                 learning_rate: float=0.1,
                 training_progress: TrainingProgress,
                 **kwargs: Any
                 ) -> Generator[TrainOptimizeStepResult, Any, None]:
        ...

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

@runtime_checkable
class ActivationFunction(Protocol):
    """
    The protocol for an activation function.
    """

    name: str
    def __call__(self, x: float) -> float:
        ...

    def derivative(self, x: float) -> float:
        ...

@runtime_checkable
class Randomizer(Protocol):
    """
    The protocol for a randomizer.
    """
    def __call__(self, shape: tuple[int, int], /) -> NPFloat2D:
        """Return a random matrix of the given shape."""
        ...

@runtime_checkable
class BuilderContext(Protocol):
        """
        The context for the builder.

        Nodes are added to the network by adding them to a layer.
        """
        net: NetProtocol
        def __init__(self, network: NetProtocol, /):
            self.net = network

        def add_layer(self, layer: 'Layer', /):
            """
            Add a layer to the network.
            """
            ...

        def add_layers(self, layers: Iterable['Layer'], /):
            """
            Add layers to the network.
            """
            ...

        def add_edge(self, edge: 'Edge', /):
            """
            Add an edge to the network.
            """
            ...
        def add_edges(self, edges: Iterable['Edge'], /):
            """
            Add edges to the network.
            """
            ...

        @property
        def layers(self) -> list['Layer']:
            """
            The layers of the network.
            """
            ...

        @property
        def nodes(self) -> Generator['Node', None, None]:
            """
            The nodes in the network.
            """
            ...

        @property
        def real_nodes(self) -> Generator['Node', None, None]:
            """
            The nodes in the network, excluding bias nodes.
            """
            ...

        @property
        def edges(self) -> Generator['Edge', None, None]:
            """
            The edges in the network.
            """
            ...

@runtime_checkable
class Builder(Protocol):
    """
    The protocol for a builder.
    """
    def __call__(self, net: BuilderContext, /,
                 *args: Any,
                 **kwargs: Any) -> None:
        ...

@runtime_checkable
class Filter(Protocol):
    """
    A filter for backpropex.
    """
    @overload
    def __call__(self, step: StepType, result: None, /,
                 **kwargs: Any) -> bool:
        """
        Prefilter a step result. If False is returned, no StepResult
        will be emitted, and no Graph will be produced.
        """
        ...
    @overload
    def __call__[T: StepType](self, step: T, result: StepResult[T], /,
                 **kwargs: Any) -> bool:
        """
        Filter a step result.

        If False is returned, no further processing on this step will be done.

        :param step: The step type.
        :param result: The step result.
        :return: True if the result is accepted, False otherwise.
        """
        ...
    @overload
    def __call__(self, step: StepType, result: Literal[False], /,
                 **kwargs: Any) -> bool:
        """
        Postfilter a step result.

        :param step: The step type.
        :param result: The step result.
        :return: False if the processing should be stopped, True otherwise.
        """
        ...

    def __call__[T: (StepTypeAny,StepType)](self, step: StepType, result: FilterArg[T], /,
                 **kwargs: Any) -> bool:
        """
        Filter a step result.

        :param step: The step type.
        :param result: The step result.
        :return: True if the result is accepted, False otherwise.
        """
        ...

class Trace(Protocol):
    """
    A trace for backpropex.
    """
    def __call__(self, step: StepType, result: StepResultAny, /, *,
                 epochs: Optional[int] = 1,
                 batch_size: Optional[int] = 1,
                 **kwargs: Any
                 ) -> None:
        """
        Trace a step result.

        :param step: The step type.
        :param result: The step result.
        """
        ...

__all__ = [
    'EvalProtocol', 'NetProtocol', 'BuilderContext',
    'TrainProtocol', 'GraphProtocol',
    'BackpropagateProtocol', 'OptimizerProtocol',
    'LossFunction', 'ActivationFunction',
    'Randomizer', 'Builder',
    'Filter', 'Trace',
]
