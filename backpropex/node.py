"""
A node in a neural network.

Nodes are connected to other nodes via edges.

Nodes have a value, which is the result of the activation function applied to the sum of the values of the nodes connected to it.
The exception i the input node, which has a value that is set externally, and may have been normalized or otherwise transformed.

Nodes are differentiated by layer type:

- Input nodes have no incoming edges.
- Hidden nodes have incoming edges from the previous layer.
- Output nodes have incoming edges from the previous layer.
- Bias nodes have a fixed value of 1.0, and are connected to all nodes in the next layer. They are only loosely coupled to a layer,
  and are displayed differently.
"""

from typing import Any, Generator, Optional, TYPE_CHECKING
import math

from backpropex.types import NPFloat1D
from backpropex.activation import ACT_Identity, ACT_ReLU, ACT_Sigmoid, ActivationFunction
if TYPE_CHECKING:
    from backpropex.layer import Layer
    from backpropex.edge import Edge


class Node:
    """
    One node in the network.
    """
    # The value of this node.
    _value: float = 0.0
    # The graphical logical position of this node in the network.
    position: tuple[float, float]
    idx: int
    layer: 'Layer'
    # The name of this node. Primarily for output nodes.
    name: Optional[str]

    activation: ActivationFunction

    loss: float = 0.0
    gradient: Optional[NPFloat1D] = None

    _edges_from: dict['Node', 'Edge']
    _edges_to: dict['Node', 'Edge']

    def addFrom(self, edge: 'Edge'):
        if self._edges_from.get(edge.to_, None) is not None:
            raise ValueError(f'Edge {edge} already exists from {self} to {edge.to_}')
        self._edges_from[edge.to_] = edge

    def addTo(self, edge: 'Edge'):
        if self._edges_to.get(edge.from_, None) is not None:
            raise ValueError(f'Edge {edge} already exists from {edge.from_} to {self}')
        self._edges_to[edge.from_] = edge

    @property
    def is_bias(self) -> bool:
        """Is this node a bias node?"""
        return False

    def __init__(self, position: tuple[float, float], /, *,
                 idx: int=-1,
                 layer: 'Layer',
                 name: Optional[str]=None,
                 activation: ActivationFunction=ACT_ReLU,
                 ):
        self.position = position
        self.idx = math.floor(idx if idx >= 0 else position[1])
        self.layer = layer
        self.name = name if name is not None else f'{layer.position}_{self.idx}'
        self._edges_from = dict()
        self._edges_to = dict()
        self.activation = activation

    @property
    def label(self) -> str:
        """The label for this node."""
        return f'{self.value:.2f}'

    @property
    def value(self) -> float:
        """The value of this node."""
        return self._value

    @value.setter
    def value(self, value: float):
        """The value of this node."""
        self._value = float(value)

    @property
    def edges(self) -> Generator['Edge', None, None]:
        """The edges connected to this node."""
        yield from self.edges_from
        yield from self.edges_to

    @property
    def edges_from(self) -> Generator['Edge', None, None]:
        """The edges from this node."""
        yield from self._edges_from.values()

    @property
    def edges_to(self) -> Generator['Edge', None, None]:
        """The edges to this node."""
        yield from self._edges_to.values()

class Input(Node):
    """
    An input node in the network.
    """
    def __init__(self, position: tuple[float, float], /, *,
                activation: ActivationFunction=ACT_Identity,
                 **kwargs: Any):
        super().__init__(position,
                         activation=activation,
                         **kwargs)

    def __repr__(self) -> str:
        return f'In[{self.idx}]={self.value:.2f}'

class Hidden(Node):
    """
    An output node in the network.--
    """
    def __init__(self, position: tuple[float, float], /,
                 **kwargs: Any):
        super().__init__(position, **kwargs)

    def __repr__(self) -> str:
        return f'Hidden[{self.layer.position},{self.idx}]={self.value:.2f}'

class Output(Node):
    """
    An output node in the network.
    """

    def __init__(self, position: tuple[float, float], /, *,
                 activation: ActivationFunction=ACT_Sigmoid,
                 name: Optional[str] = None,
                **kwargs: Any
                 ):
        super().__init__(position,
                        activation=activation,
                         **kwargs)
        self.name = name

    def __repr__(self) -> str:
        if self.name is None:
            return f'Out[{self.idx}]={self.value:.2f}'
        return f'Out.{self.name}={self.value:.2f}'

class Bias(Node):
    """
    A bias node in the network.
    """
    def __init__(self, position: tuple[float, float], /, **kwargs: Any):
        super().__init__(position,
                         activation=ACT_Identity,
                         **kwargs)
    @property
    def is_bias(self) -> bool:
        """Is this node a bias node?"""
        return True

    @property
    def value(self) -> float:
        """The value of this node."""
        return 1.0

    @value.setter
    def value(self, value: float):
        """The value of this node."""
        raise ValueError('Cannot set the value of a bias node.')

    @property
    def label(self) -> str:
        """The label for this node."""
        return "1"

    def __repr__(self):
        return "<1>"

__all__ = ['Node', 'Input', 'Hidden', 'Output', 'Bias']
