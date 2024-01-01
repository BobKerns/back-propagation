from collections.abc import Sequence
from typing import Callable, Optional
import math
import random

import numpy as np
from networkx import DiGraph, draw_networkx
from matplotlib import colormaps

type ActivationFunction = Callable[[float], float]

def ReLU(x: float) -> float:
    """The standard ReLU activation function."""
    return max(0.0, x)

def softmax(x: float) -> float:
    """The standard softmax activation function."""
    return 1.0 / (1.0 + math.exp(-x))

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
    def __init__(self, position: tuple[int, int], /, *,
                 activation_function: ActivationFunction=ReLU):
        self.position = position
        # The function which calculates this node's value
        self.activation = activation_function

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

    def __init__(self, nodes: int, /, *,
                 position: int = 0,
                 activation_function: ActivationFunction=ReLU,
                 max_layer_size: int=10):
        """
        A set of nodes constituting one layer in a neural network.

        Inaddition to the specified number of nodes, a bias node is added to the layer.

        :param nodes: The number of nodes in this layer.
        :param activation_function: The activation function for each node in this layer.
        :param max_layer_size: The maximum number of nodes in any layer, for centering the nodes.
        """
        self.position = position
        offset = (max_layer_size - nodes) / 2
        self.bias = Node((self.position, offset), activation_function=activation_function)
        def node(position: int):
            pos = (self.position, position + 1 + offset)
            return Node(pos, activation_function=activation_function)
        self.nodes = [self.bias] + [node(position) for position in range(nodes)]

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
    def __init__(self, *layers: int,
                 loss_function: LossFunction=LossFunction(),
                 activation_functions: Sequence[ActivationFunction]=None):
        """
        A neural network.

        :param layers: The number of nodes in each layer.
        :param loss_function: The loss function for this network.
        :param activation_functions: The activation function for each layer.
        """
        if activation_functions is None:
            activation_functions = [ReLU] * (len(layers) - 1) + [softmax]
        self.layers = [
            Layer(nodes, position=idx, activation_function=activation_function)
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
            for (nidx, next_node) in enumerate(next.nodes):
                edge = Edge(node, next_node, initial_weight=w[nidx][pidx])
                self.graph.add_edge(node, next_node, edge=edge)

    @property
    def labels(self):
        return {node: node.label for node in self.graph.nodes}

    @property
    def positions(self):
        def place(node: Node):
            pos = node.position
            return (pos[0] * 100, pos[1] * 100)
        return {node: place(node) for node in self.graph.nodes}

    @property
    def node_colors(self):
        return [node.value for node in self.graph.nodes]

    @property
    def edge_colors(self):
        return [d['edge'].weight for (f, t, d) in self.graph.edges(data=True)]

    def draw(self):
        draw_networkx(self.graph, self.positions,
                      labels=self.labels,
                      node_size=1000,
                      node_color=self.node_colors,
                      vmin=-1, vmax=1,
                      cmap=colormaps.get_cmap('coolwarm'),
                      edge_color=self.edge_colors,
                      edge_cmap=colormaps.get_cmap('coolwarm'),
                      edge_vmin=-1, edge_vmax=1)
