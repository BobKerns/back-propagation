from collections.abc import Sequence
from enum import Enum
from typing import Callable, Optional
from functools import cached_property
import math

import numpy as np
from networkx import DiGraph, draw_networkx
from matplotlib import colormaps
import matplotlib.pyplot as plt

type ActivationFunction = Callable[[float], float]

def ReLU(x: float) -> float:
    """The standard ReLU activation function."""
    return max(0.0, x)

def softmax(x: float) -> float:
    """The standard softmax activation function."""
    return 1.0 / (1.0 + math.exp(-x))

class LayerType(Enum):
    """
    The type of a layer in a neural network.
    """
    Input = 0
    Hidden = 1
    Output = 2

class LossFunction:
    """
    A cost function for a neural network.
    """
    def __init__(self, /):
        pass
    def evaluate(self, actual: np.array, expected: np.array) -> float:
        """
        Evaluate the cost function for a given set of actual and expected values.
        """
        raise NotImplemented

class Node:
    """
    One node in the network.
    """
    # The value of this node.
    value: float = 0.0
    # The activation function for this node.
    activation: ActivationFunction
    # The graphical logical position of this node in the network.
    position: tuple[float, float]

    is_bias: bool = False
    def __init__(self, position: tuple[int, int], /, *,
                 activation_function: ActivationFunction=ReLU,
                 is_bias: bool=False):
        self.position = position
        # The function which calculates this node's value
        self.activation = activation_function
        self.is_bias = is_bias
        if is_bias:
            self.value = 1.0

    @property
    def label(self) -> setattr:
        return f'{self.value:.2f}'

class Edge:
    """
    `Edge` connects two nodes in the network.

    It holds the weight of the connection between the two nodes.

    The weight is used to calculate a gradient for a weight with respect to the cost function.
    """
    # The weight of this edge.
    weight: float
    previous: Node
    next: Node
    def __init__(self, previous, next, /, *,
                 initial_weight: float=0.0):
        self.previous = previous
        self.next = next
        self.weight = initial_weight

class Layer:
    # Label for this layer
    label: str
    # Nodes in this layer
    nodes: Sequence[Node]
    # Bias node for this layer (always the first node)
    bias: Node

    layer_type: LayerType

    def __init__(self, nodes: int, /, *,
                 position: int = 0,
                 activation_function: ActivationFunction=ReLU,
                 max_layer_size: int=10,
                 layer_type: LayerType=LayerType.Hidden
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
        positions = iter(range(0, nodes + 1))
        match layer_type:
            case LayerType.Input | LayerType.Hidden:
                offset = (max_layer_size - nodes) / 2
            case LayerType.Output:
                offset = (max_layer_size - nodes + 1) / 2
        def node(**kwargs):
            position = next(positions)
            pos = (self.position, position + offset)
            return Node(pos, activation_function=activation_function, **kwargs)
        bias = []
        match layer_type:
            case LayerType.Input | LayerType.Hidden:
                self.bias = node(is_bias=True)
                bias = [self.bias]
        self.nodes = bias + [node() for _ in range(nodes)]

    @property
    def real_nodes(self):
        match self.layer_type:
            case LayerType.Input | LayerType.Hidden:
                return self.nodes[1:]
            case LayerType.Output:
                return self.nodes

    @property
    def values(self):
        return np.array((n.value for n in self.nodes))

    @values.setter
    def values(self, values: np.array):
        for node, value in zip(self.nodes, values):
            node.value = value

class Network:
    """
    A neural network.
    """
    layers: Sequence[Layer]
    loss_function: Callable[[float], float]
    graph: DiGraph = DiGraph()
    margin: float
    max_layer_size: int
    def __init__(self, *layers: int,
                 loss_function: LossFunction=LossFunction(),
                 activation_functions: Sequence[ActivationFunction]=None,
                 margin: float=100.0):
        """
        A neural network.

        :param layers: The number of nodes in each layer.
        :param loss_function: The loss function for this network.
        :param activation_functions: The activation function for each layer.
        """
        self.margin = margin
        if activation_functions is None:
            activation_functions = [ReLU] * (len(layers) - 1) + [softmax]
        self.max_layer_size = max(layers)
        def layer_type(idx: int):
            match idx:
                case 0:
                    return LayerType.Input
                case _ if idx == len(layers) - 1:
                    return LayerType.Output
                case _:
                    return LayerType.Hidden

        self.layers = [
            Layer(nodes,
                  position=idx,
                  activation_function=activation_function,
                  max_layer_size=self.max_layer_size,
                  layer_type=layer_type(idx))
            for nodes, activation_function, idx
            in zip(layers, activation_functions, range(len(layers)))
        ]
        self.loss_function = loss_function
        # Label the layers
        self.layers[0].label = 'Input'
        self.layers[-1].label = 'Output'
        for idx, layer in enumerate(self.layers[1:-1]):
            layer.label = f'Hidden_{idx}'
        # Connect the layer nodes
        prev: Layer = self.layers[0]
        for layer in self.layers[1:]:
            self.connect_layers(prev, layer)
            prev = layer

    def connect_layers(self, prev: Layer, next: Layer):
        """
        Connect two layers in the network.
        """
        psize = len(prev.nodes)
        nsize = len(next.nodes)
        # Initialize the edge weights by He-et-al initialization
        w = np.random.randn(nsize, psize) * np.sqrt(2 / psize)
        for (pidx, node) in enumerate(prev.nodes):
            for (nidx, next_node) in enumerate(next.real_nodes):
                edge = Edge(node, next_node, initial_weight=w[nidx][pidx])
                self.graph.add_edge(node, next_node, edge=edge)

    @property
    def labels(self):
        return {node: node.label for node in self.graph.nodes}

    @cached_property
    def positions(self):
        def place(node: Node):
            pos = node.position
            return (pos[0] * 100, pos[1] * 100 + self.margin)
        return {node: place(node) for node in self.graph.nodes}

    @property
    def node_colors(self):
        return [0 if node.is_bias else node.value for node in self.graph.nodes]

    @property
    def edge_colors(self):
        return [d['edge'].weight for (f, t, d) in self.graph.edges(data=True)]

    @property
    def edges(self):
        return [(f, t, d['edge']) for (f, t, d) in self.graph.edges(data=True)]

    def draw(self):
        """
        Draw the network using matplotlib.
        """
        plt.close()
        fig, ax = plt.subplots()
        ax.set_ylim(25, self.max_layer_size * 100 + self.margin + 50)
        draw_networkx(self.graph, self.positions,
                            labels=self.labels,
                            node_size=1000,
                            node_color=self.node_colors,
                            vmin=-1, vmax=1,
                            cmap=colormaps.get_cmap('coolwarm'),
                            edgecolors=['blue' if node.is_bias else 'black' for node in self.graph.nodes],
                            edge_color=self.edge_colors,
                            edge_cmap=colormaps.get_cmap('coolwarm'),
                            edge_vmin=-1, edge_vmax=1,
                            ax=ax)
        positions = self.positions
        for f, t, edge in self.edges:
            loc1 = positions[f]
            loc2 = positions[t]
            loc = (loc1[0] * 0.8 + loc2[0] * 0.2), (loc1[1] * 0.75 + loc2[1] * 0.25 - 0)
            ax.annotate(f'{edge.weight:.2f}',loc)
        for layer in self.layers:
            ax.annotate(layer.label, (layer.position * 100 - 10, 50))
        plt.show()
