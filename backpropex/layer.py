"""
Neural Networ Layer
"""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from backpropex.protocols import Randomizer
from backpropex.randomizer import HeEtAl

from backpropex.types import FloatSeq, LayerType, NPFloat1D, NPFloat2D, NPObject2D
from backpropex.activation import ACT_Identity, ACT_ReLU, ActivationFunction
from backpropex.node import Bias, Hidden, Input, Node, Output

class Layer:
    """One layer in a neural network."""
    # Label for this layer
    label: str
    # Nodes in this layer
    nodes: Sequence[Node]

    edges_to: NPObject2D
    edges_from: NPObject2D
    # Bias node for this layer (always the first node)
    bias: Node

    layer_type: LayerType

    # The position of this layer in the network, with input = 0
    idx: int

    # The activation function for nodes in this layer
    _activation_fn: ActivationFunction
    _activation_fn_ufunc: np.ufunc
    _activation_fn_derivative: np.ufunc
    @property
    def activation_fn(self):
        return self._activation_fn

    @activation_fn.setter
    def activation_fn(self, fn: ActivationFunction):
        self._activation_fn = fn
        self._activation_fn_ufunc = np.frompyfunc(fn, 1, 1)
        self._activation_fn_derivative = np.frompyfunc(fn.derivative, 1, 1)

    @property
    def activation_fn_derivative(self):
        return self._activation_fn_derivative

    _values: NPFloat1D

    weights: NPFloat2D

    gradient: NPFloat1D

    loss: NPFloat1D

    loss_delta: NPFloat1D

    weight_delta: NPFloat1D

    def __init__(self, nodes: int, prev_nodes: Optional[int] = None, /, *,
                 idx: int = 0,
                 activation: ActivationFunction=ACT_ReLU,
                 max_layer_size: int=10,
                 layer_type: LayerType=LayerType.Hidden,
                 names: Optional[Sequence[str]]=None,
                 randomizer: Randomizer = HeEtAl(),
                 ):
        """
        A set of nodes constituting one layer in a neural network.

        In addition to the specified number of nodes, a bias node is added to the layer.

        :param nodes: The number of nodes in this layer.
        :param prev_nodes: The number of nodes in the previous layer, if any
        :param activation_function: The activation function for each node in this layer.
        :param max_layer_size: The maximum number of nodes in any layer, for centering the nodes.
        """
        self.idx = idx
        offset = (max_layer_size - nodes) / 2
        self.layer_type = layer_type
        self.activation_fn = activation
        positions = iter(range(0, nodes + 1))
        match layer_type:
            case LayerType.Input | LayerType.Hidden:
                offset = (max_layer_size - nodes) / 2
                lsize = nodes + 1
                lnames = [f'Bias_{idx}', *(names or ((None,)* nodes))]
            case LayerType.Output:
                offset = (max_layer_size - nodes + 2) / 2
                lsize = nodes
                lnames = names or ((None,) * nodes)
        def node(idx: int = 0, is_bias: bool = False, **kwargs: Any):
            """Construct a suitable node for this layer."""
            position = next(positions)
            pos = (float(self.idx), position + offset)
            if (is_bias):
                return Bias(pos, layer=self)
            name = None if names is None else lnames[idx] or f'{self.idx}_{idx}'
            match idx, layer_type:
                case 0, LayerType.Input|LayerType.Hidden:
                    self.bias = Bias(pos, layer=self, idx=idx, name=name, **kwargs)
                    return self.bias
                case _, LayerType.Input:
                    return Input(pos, layer=self, idx=idx, activation=ACT_Identity, name=name, **kwargs)
                case _, LayerType.Output:
                    return Output(pos, layer=self, idx=idx, activation=activation, name=name, **kwargs)
                case _, LayerType.Hidden:
                    return Hidden(pos, layer=self, idx=idx, activation=activation, name=name, **kwargs)
        self._values = np.zeros(lsize, np.float_)
        if prev_nodes is not None:
            self.weights = randomizer((prev_nodes + 1, lsize))
            self.weights[:, 0] = 0.0 # No incoming weights to the bias node
        self.nodes = [node(idx) for idx in range(lsize)]
        self.gradient = np.zeros(len(self.nodes), np.float_)
        self.loss = np.zeros(len(self.nodes), np.float_)
        self.loss_delta = np.zeros(len(self.nodes), np.float_)
        self.weight_delta = np.zeros(len(self.nodes), np.float_)

    @property
    def real_nodes(self):
        match self.layer_type:
            case LayerType.Input | LayerType.Hidden:
                return self.nodes[1:]
            case LayerType.Output:
                return self.nodes

    def activation(self, values: NPFloat1D, /) -> NPFloat1D:
        """The values of the activation function for the nodes in this layer."""
        return self._activation_fn_ufunc(values)

    @property
    def values(self):
        """The values of the nodes in this layer."""
        match self.layer_type:
            case LayerType.Input | LayerType.Hidden:
                return self._values[1:]
            case LayerType.Output:
                return self._values

    @values.setter
    def values(self, values: FloatSeq):
        """ Set the values of the nodes in this layer."""
        match self.layer_type:
            case LayerType.Input | LayerType.Hidden:
                self._values[1:] = values
            case LayerType.Output:
                self._values = np.array(values, dtype=np.float_)

    def value(self, idx: int):
        """Get the value of a node in this layer."""
        return self._values[idx]

    def set_value(self, idx: int, value: float):
        """Set the value of a node in this layer."""
        self._values[idx] = value

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
        return f'{self.layer_type} {self.activation_fn.name}({len(self.nodes)})'

__all__ = ['Layer']
