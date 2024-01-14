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

from collections.abc import Sequence
from typing import Any, Optional, TYPE_CHECKING, cast
import math

from backpropex.types import NPFloat1D
from backpropex.activation import ACT_Identity, ActivationFunction
if TYPE_CHECKING:
    from backpropex.layer import Layer
from backpropex.edge import Edge

class Node:
    """
    One node in the network.
    """
    # The graphical logical position of this node in the network.
    position: tuple[float, float]
    id: int
    idx: int
    layer: 'Layer'
    # The name of this node. Primarily for output nodes.
    name: Optional[str]

    loss: float = 0.0
    gradient: Optional[NPFloat1D] = None

    @property
    def is_bias(self) -> bool:
        """Is this node a bias node?"""
        return False

    def __init__(self, position: tuple[float, float], /, *,
                 idx: int=-1,
                 layer: 'Layer',
                 name: Optional[str]=None,
                 **kwargs: Any
                 ):
        self.position = position
        self.idx = math.floor(idx if idx >= 0 else position[1])
        self.layer = layer
        self.name = name if name is not None else f'{layer.idx}_{self.idx}'

    @property
    def label(self) -> str:
        """The label for this node."""
        return f'{self.value:.2f}'

    @property
    def value(self) -> float:
        """The value of this node."""
        return self.layer.value(self.idx)

    @value.setter
    def value(self, value: float):
        """The value of this node."""
        return self.layer.set_value(self.idx, value)

    @property
    def edges_from(self) -> Sequence[Edge]:
        """The edges from this node."""
        if hasattr(self.layer, 'edges_from'):
            ar = self.layer.edges_from[self.idx, :]
            return cast(Sequence[Edge], ar[ar != None]) # type: ignore
        return []

    @property
    def edges_to(self) -> Sequence[Edge]:
        """The edges to this node."""

        if hasattr(self.layer, 'edges_to'):
            ar = self.layer.edges_to[self.idx, :]
            return cast(Sequence[Edge], ar[ar != None]) # type: ignore
        return []

    @property
    def activation(self) -> ActivationFunction:
        """The activation function for this node."""
        return self.layer.activation_fn


class Input(Node):
    """
    An input node in the network.
    """
    def __init__(self, position: tuple[float, float], /,
                 **kwargs: Any):
        super().__init__(position,
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
        return f'Hidden[{self.layer.idx},{self.idx}]={self.value:.2f}'

class Output(Node):
    """
    An output node in the network.
    """

    def __init__(self, position: tuple[float, float], /, *,
                 name: Optional[str] = None,
                **kwargs: Any
                 ):
        super().__init__(position,
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

    @property
    def activation(self) -> ActivationFunction:
        """The activation function for this node."""
        return ACT_Identity

    def __repr__(self):
        return "<1>"

__all__ = ['Node', 'Input', 'Hidden', 'Output', 'Bias']
