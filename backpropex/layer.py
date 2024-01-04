"""
Neural Networ Layer
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Optional

from backpropex.types import FloatSeq, LayerType
from backpropex.activation import ACT_ReLU, ActivationFunction
from backpropex.node import Bias, Hidden, Input, Node, Output


class LayerType(StrEnum):
    """
    The type of a layer in a neural network.
    """
    Input = "Input"
    Hidden = "Hidden"
    Output = "Output"

class Layer:
    """One layer in a neural network."""
    # Label for this layer
    label: str
    # Nodes in this layer
    nodes: Sequence[Node]
    # Bias node for this layer (always the first node)
    bias: Node

    layer_type: LayerType

    # The position of this layer in the network, with input = 0
    position: int

    # The activation function for nodes in this layer
    # We record it here so we can display it in the graph
    activation: ActivationFunction

    def __init__(self, nodes: int, /, *,
                 position: int = 0,
                 activation: ActivationFunction=ACT_ReLU,
                 max_layer_size: int=10,
                 layer_type: LayerType=LayerType.Hidden,
                 names: Optional[Sequence[str]]=None,
                 ):
        """
        A set of nodes constituting one layer in a neural network.

        Inaddition to the specified number of nodes, a bias node is added to the layer.

        :param nodes: The number of nodes in this layer.
        :param activation_function: The activation function for each node in this layer.
        :param max_layer_size: The maximum number of nodes in any layer, for centering the nodes.
        """
        self.position = position
        offset = (max_layer_size - nodes) / 2
        self.layer_type = layer_type
        self.activation = activation
        positions = iter(range(0, nodes + 1))
        match layer_type:
            case LayerType.Input | LayerType.Hidden:
                offset = (max_layer_size - nodes) / 2
            case LayerType.Output:
                offset = (max_layer_size - nodes + 2) / 2\

        def node(idx: Optional[int] = None, is_bias: bool = False, **kwargs):
            """Construct a suitable node for this layer."""
            position = next(positions)
            pos = (self.position, position + offset)
            if (is_bias):
                return Bias(pos, layer=self)
            name = None if names is None else names[idx]
            match layer_type:
                case LayerType.Input:
                    return Input(pos, layer=self, idx=idx, name=name, **kwargs)
                case LayerType.Output:
                    return Output(pos, layer=self, idx=idx, activation=activation, name=name, **kwargs)
                case LayerType.Hidden:
                    return Hidden(pos, layer=self, idx=idx, activation=activation, name=name, **kwargs)
                case _:
                    raise ValueError(f'Unknown layer type: {layer_type}')
        bias = []
        match layer_type:
            case LayerType.Input | LayerType.Hidden:
                self.bias = node(is_bias=True)
                bias = [self.bias]
        self.nodes = bias + [node(idx) for idx in range(nodes)]

    @property
    def real_nodes(self):
        match self.layer_type:
            case LayerType.Input | LayerType.Hidden:
                return self.nodes[1:]
            case LayerType.Output:
                return self.nodes

    @property
    def values(self):
        return [n.value for n in self.real_nodes]

    @values.setter
    def values(self, values: FloatSeq):
        for node, value in zip(self.real_nodes, values):
            node.value = value

    def __getitem__(self, idx: int|str):
        """Get a node by index or name."""
        if isinstance(idx, str):
            for node in self.nodes:
                if node.name == idx:
                    return node
            raise ValueError(f'No node with name {idx}')
        return self.real_nodes[idx]

    def __len__(self):
        """The number of real nodes in this layer."""
        return len(self.real_nodes)

    def __repr__(self):
        return f'{self.layer_type} {self.activation.name}({len(self.nodes)})'

__all__ = ['Layer']
