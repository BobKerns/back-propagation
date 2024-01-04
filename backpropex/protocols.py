"""
Protocols to allow the various classes to collaborate.
"""

from typing import Sequence, Optional, runtime_checkable, Protocol, TYPE_CHECKING, Any, Generator
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


@runtime_checkable
class EvalProtocol(Protocol):
    def __call__(self, input: FloatSeq, /, *,
                label: Optional[str] = None
                ) -> Generator[EvalStepResultAny|InitStepResult, Any, None]:
        ...
    net: 'NetProtocol'

@runtime_checkable
class NetProtocol(EvalProtocol, Protocol):
    net: 'NetProtocol'
    graph: DiGraph = DiGraph()
    layers: Sequence['Layer']
    max_layer_size: int
    name: str

    # Progress information for drawing the network
    # The layer that is currently being evaluated
    active_layer: Optional['Layer'] = None
    active_message: Optional[str] = None

    def __call__(self, input: FloatSeq, /, *,
                label: Optional[str] = None
                ) -> Generator[EvalStepResultAny, Any, None]:
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
    def __call__(self, data: TrainingData, /, *,
              epochs: int=1000,
              learning_rate: float=0.1
              ) -> Generator[TrainStepResultAny|InitStepResult, Any, None]:
        ...
    net: NetProtocol
