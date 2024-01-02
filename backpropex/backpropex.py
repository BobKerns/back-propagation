from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Generator, Optional
from functools import cached_property
import math
from matplotlib.colors import Colormap

import numpy as np
from networkx import DiGraph, draw_networkx, draw_networkx_nodes, draw_networkx_edges, draw_networkx_labels
from matplotlib import colormaps
import matplotlib.pyplot as plt

type ActivationFunction = Callable[[float], float]
type ActivationDerivative = Callable[[float], float]

@dataclass
class Activation:
    """
    An activation function and its derivative.
    """
    name: str
    function: ActivationFunction
    derivative: ActivationDerivative
    def __call__(self, x: float) -> float:
        return self.function(x)

def ReLU(x: float) -> float:
    """The standard ReLU activation function."""
    return max(0.0, x)

def ReLUDerivative(x: float) -> float:
    """The derivative of the standard ReLU activation function."""
    return 1.0 if x > 0.0 else 0.0

ACT_ReLU = Activation('ReLU', ReLU, ReLUDerivative)

def softmax(x: float) -> float:
    """The standard softmax activation function."""
    return 1.0 / (1.0 + math.exp(-x))

def softmax_derivative(x: float) -> float:
    """The derivative of the standard softmax activation function."""
    return softmax(x) * (1.0 - softmax(x))

ACT_Softmax = Activation('Softmax', softmax, softmax_derivative)

def sigmoid(x: float) -> float:
    """The standard sigmoid activation function."""
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x: float) -> float:
    """The derivative of the standard sigmoid activation function."""
    return sigmoid(x) * (1.0 - sigmoid(x))

ACT_Sigmoid = Activation('Sigmoid', sigmoid, sigmoid_derivative)

class LayerType(StrEnum):
    """
    The type of a layer in a neural network.
    """
    Input = "Input"
    Hidden = "Hidden"
    Output = "Output"

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
    activation: Activation
    # The graphical logical position of this node in the network.
    position: tuple[float, float]
    # The name of this node. Primarily for output nodes.
    name: Optional[str]

    is_bias: bool = False
    def __init__(self, position: tuple[int, int], /, *,
                 is_bias: bool=False,
                 name: Optional[str]=None):
        self.position = position
        if is_bias:
            self.value = 1.0
        self.name = name

    @property
    def label(self) -> setattr:
        """The label for this node."""
        if self.is_bias:
            return "1"
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

    @property
    def label(self):
        return f'{self.weight:.2f}'

    def __repr__(self):
        return f'Edge({self.previous} -> {self.next})'

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
    activation: Activation

    def __init__(self, nodes: int, /, *,
                 position: int = 0,
                 activation: Activation=ACT_ReLU,
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
                offset = (max_layer_size - nodes + 2) / 2
        def node(**kwargs):
            position = next(positions)
            pos = (self.position, position + offset)
            name = None if names is None else names[idx]
            return Node(pos, activation=activation, name=name, **kwargs)
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
        return [n.value for n in self.real_nodes]

    @values.setter
    def values(self, values: list[float]):
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
                 activations: Sequence[Activation]=None,
                 margin: float=0.13,
                 input_names: Optional[Sequence[str]]=None,
                 output_names: Optional[Sequence[str]]=None,
                 ):
        """
        A neural network.

        :param layers: The number of nodes in each layer.
        :param loss_function: The loss function for this network.
        :param activation_functions: The activation function for each layer.
        """
        self.margin = margin
        if activations is None:
            activations = [ACT_ReLU] * (len(layers) - 1) + [ACT_Sigmoid]
        self.max_layer_size = max(layers)
        def layer_type(idx: int):
            match idx:
                case 0:
                    return LayerType.Input
                case _ if idx == len(layers) - 1:
                    return LayerType.Output
                case _:
                    return LayerType.Hidden

        def node_names(ltype: LayerType):
            if ltype == LayerType.Input:
                if input_names is None:
                    return [f'In[{idx}]' for idx in range(layers[0])]
                return input_names
            if ltype == LayerType.Output:
                if output_names is None:
                    return [f'Out[{idx}]' for idx in range(layers[-1])]
                return output_names
            return None
        def make_layer(idx: int, nodes: int, activation: Activation):
            ltype = layer_type(idx)
            return Layer(nodes,
                    position=idx,
                    activation=activation,
                    max_layer_size=self.max_layer_size,
                    names=node_names(ltype),
                    layer_type=ltype)
        self.layers = [
            make_layer(idx, nodes, activation)
            for nodes, activation, idx
            in zip(layers, activations, range(len(layers)))
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
        return {n: n.label for n in self.graph.nodes}

    @cached_property
    def positions(self):
        """Compute the positions of the nodes in the graph."""
        xscale = 1.0 / len(self.layers)
        yscale = 1.0 / (self.max_layer_size + 1)
        def place(node: Node):
            pos = node.position
            xpos = pos[0] + 0.5 if node.is_bias else pos[0]
            ypos = 0 if node.is_bias else pos[1]
            return (xpos * xscale+ 0.1, ypos * yscale + self.margin)
        return {node: place(node) for node in self.graph.nodes}

    @property
    def node_colors(self):
        return [node.value for node in self.graph.nodes]

    @property
    def edge_colors(self):
        return [edge.weight for (f, t, edge) in self.graph.edges(data='edge')]

    @property
    def edges(self):
        return [edge for (f, t, edge) in self.graph.edges(data='edge')]

    def draw(self, /, *, label: str="Initial State"):
        """
        Draw the network using matplotlib.
        """
        plt.close()
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_autoscale_on(False)
        #top =self.max_layer_size * 250 + self.margin + 150
        #ax.set_ylim(25, top)
        coolwarm: Colormap = colormaps.get_cmap('coolwarm'),
        coolwarm = coolwarm[0]
        def draw_nodes(nodelist, /, *,
                       node_size=1000,
                       edgecolors='black',
                       font_color='black',
                       **kwargs):
            draw_networkx_nodes(self.graph, self.positions,
                                nodelist=nodelist,
                                node_size=node_size,
                                node_color=[n.value for n in nodelist],
                                vmin=-2.5, vmax=2.5,
                                cmap=coolwarm,
                                edgecolors=edgecolors,
                                label=label,
                                ax=ax,
                                **kwargs)
            text_y_offset = 0.006
            for node in nodelist:
                pos_x, pos_y = self.positions[node]
                if node.is_bias:
                    pos = (pos_x - 0.004, pos_y - text_y_offset)
                else:
                    pos = (pos_x - 0.013, pos_y - text_y_offset)
                ax.annotate(node.label, pos,
                            color=font_color)
                if node.name is not None:
                    ax.annotate(node.name, (pos_x - 0.013, pos_y - text_y_offset - 0.04),
                                color=font_color)
        regular_nodes = [node for node in self.graph.nodes if not node.is_bias]
        bias_nodes = [node for node in self.graph.nodes if node.is_bias]
        # Draw the regular nodes first
        draw_nodes(regular_nodes)
        # Draw the bias nodes distinctively differently.
        draw_nodes(bias_nodes,
                   node_shape='s',
                   node_size=500,
                   linewidths=2.0,
                   edgecolors='green',
                   font_color='green')
        draw_networkx_edges(self.graph, self.positions,
                            edge_color=self.edge_colors,
                            edge_cmap=coolwarm,
                            edge_vmin=-1, edge_vmax=1,
                            ax=ax)
        ax.set_title(label)
        positions = self.positions
        # Offsets along the edge to avoid overlapping labels
        shifts = (-0.05, 0.05, 0.075)
        for idx, edge in enumerate(self.edges):
            loc1 = positions[edge.previous]
            loc2 = positions[edge.next]
            # Choose the shift for the label to avoid conflicts
            shift = shifts[idx % len(shifts)]
            loc_x = loc1[0] *(0.8 + shift) + loc2[0] * (0.2 - shift)
            loc_y = loc1[1] * (0.75 + shift) + loc2[1] * (0.25 - shift)
            loc = loc_x, loc_y
            #  Compute the color for the label based on the edge weight
            #  This matches how the edge is colored
            weight = edge.weight
            color = coolwarm((weight + 1) / 2)
            ax.annotate(edge.label, loc, color=color)
        # Label the layers on the graph
        layer_x_offset = 0.085
        layer_y_offset = 0.05
        x_scale = 1.0 / len(self.layers)
        for layer in self.layers:
            ax.annotate(layer.label, (layer.position * x_scale + layer_x_offset, layer_y_offset))
        for layer in self.layers[0:-1]:
            ax.annotate('Bias', ((layer.position + 0.5) * x_scale + layer_x_offset, layer_y_offset),
                        color='green')
        for layer in self.layers[1:]:
            ax.annotate(layer.activation.name, (layer.position * x_scale + layer_x_offset, layer_y_offset - 0.025))
        plt.show()

    def show(self, label: str):
        self.draw(label=label)
        return label
    def evaluate(self, input: np.array, /, *,
                 epoch: Optional[int] = None
                 ) -> Generator[np.ndarray[Any], Any, np.ndarray[Any]]:
        """
        Evaluate the network for a given input.
        """
        layer = self.layers[0]
        layer.values = input
        epoch_label = '' if epoch is None else f'Epoch: {epoch} '
        yield self.show(label=f'{epoch_label}{layer.label}: {''.join(map(repr, layer.values))}')
        for layer in self.layers[1:]:
            for node in layer.real_nodes:
                node.value = sum(edge.weight * edge.previous.value for f, t, edge in self.graph.in_edges(node, data='edge'))
                node.value = node.activation(node.value)
            yield self.show(label=f'{epoch_label}Forward: {layer.label}')
        yield self.output_layer.values

    def train_one(self, input: np.array, expected: np.array, /, *,
                  epoch:int = 0
                  ) -> Generator[np.ndarray[Any], Any, np.ndarray[Any]]:
        """
        Train the network for a given input and expected output.
        """
        # Forward pass
        yield from self.evaluate(input, epoch=epoch)
        # Backward pass
        layer = self.layers[-1]
        yield self.show(label=f'Epoch: {epoch} Backward: {layer.label}')
        for layer in reversed(self.layers[0:-1]):
            for node in layer.real_nodes:
                node.value = sum(edge.weight * edge.next.value for f, t, edge in self.graph.out_edges(node, data='edge'))
                node.value *= node.value

    def train(self, data: np.array, /, *, epochs: int=1000, learning_rate: float=0.1):
        """
        Train the network for a given set of data.
        """
        for epoch in range(epochs):
            for input, expected in data:
                yield from self.train_one(input, expected, epoch=epoch)
            yield self.show(label=f'Epoch: {epoch}')
    @property
    def input_layer(self):
        """The input layer of this network."""
        return self.layers[0]

    @property
    def output_layer(self):
        """The output layer of this network."""
        return self.layers[-1]

    @property
    def hidden_layers(self):
        """The hidden layers of this network."""
        return self.layers[1:-1]

    def __getitem__(self, idx: int):
        """Get a layer by index."""
        return self.layers[idx]

    def __len__(self):
        """The number of layers in this network."""
        return len(self.layers)

    def __repr__(self):
        return f'Network({",".join((str(len(l)) for l in self.layers))})'
