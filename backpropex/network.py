
from collections import namedtuple
from contextlib import contextmanager
from functools import cached_property
from typing import Any, Generator, Optional, Sequence
import re
import math
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import _colormaps
from matplotlib.colors import Colormap
from matplotlib.patches import FancyBboxPatch

from networkx import DiGraph, draw_networkx_edges, draw_networkx_nodes
import numpy as np

from backpropex.types import NPArray
from backpropex.activation import ACT_ReLU, ACT_Sigmoid, ActivationFunction
from backpropex.edge import Edge
from backpropex.layer import Layer, LayerType
from backpropex.loss import LossFunction, MeanSquaredError
from backpropex.node import Node

def _ids():
    """Generate unique ids."""
    idx = 0
    while True:
        yield idx
        idx += 1

ids = _ids()

# Constants for drawing the network
layer_x_offset = 0.085
layer_y_offset = 0.05 # Main row
layer_y2_offset = layer_y_offset - 0.025 # second row, below main row
layer_y0_offset = layer_y_offset + 0.025 # zeroth row, above main row

def plen(p: tuple[float, float], n: tuple[float, float]) -> float:
    """{Pythagorhean length of a line. expressed as two points"""
    x2 = (p[0] - n[0]) ** 2.0
    y2 = (p[1] - n[1]) ** 2.0
    return math.sqrt(x2 + y2)

class Network:
    """
    A neural network.
    """
    layers: Sequence[Layer]
    loss_function: LossFunction
    graph: DiGraph = DiGraph()
    margin: float
    max_layer_size: int
    name: str
    input_type: type[namedtuple]
    output_type: type[namedtuple]
    expected: Optional[Sequence[float]] = None
    xscale: float
    yscale: float
    # The layer that is currently being evaluated
    active_layer: Optional[Layer] = None
    active_message: Optional[str] = None

    def __init__(self, *layers: int,
                 name: Optional[str] = None,
                 loss_function: LossFunction=MeanSquaredError,
                 activations: Sequence[ActivationFunction]=None,
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
        self.name = name if name is not None else f'Network_{next(ids)}'

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
        def make_layer(idx: int, nodes: int, activation: ActivationFunction):
            ltype = self.layer_type(idx, len(layers))
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
        # Label the layers
        self.layers[0].label = 'Input'
        self.layers[-1].label = 'Output'
        for idx, layer in enumerate(self.layers[1:-1]):
            layer.label = f'Hidden[{idx}]'
        # Connect the layer nodes
        prev: Layer = self.layers[0]
        for layer in self.layers[1:]:
            self.connect_layers(prev, layer)
            prev = layer

        self.loss_function = loss_function

        self.xscale = 1.0 / (len(self.layers) + 0.4)
        self.yscale = 1.0 / (self.max_layer_size + 1)

        # We are all set up, so let's define our input and output.
        def sanitize(name: str):
            return '_'.join((s for s in re.split(r'[^a-zA-Z0-9_]+', name) if s != ''))
        print([sanitize(node.name) for node in self.input_layer.real_nodes])
        self.input_type = namedtuple(f'{sanitize(self.name)}_input',
                                       [sanitize(node.name) for node in self.input_layer.real_nodes])
        self.output_type = namedtuple(f'{sanitize(self.name)}_output',
                                       [sanitize(node.name) for node in self.output_layer.real_nodes])
    def layer_type(self, idx: int, nlayers: int = None):
        if nlayers is None:
            nlayers = len(self.layers)
        match idx:
            case 0:
                return LayerType.Input
            case _ if idx == nlayers - 1:
                return LayerType.Output
            case _:
                return LayerType.Hidden

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
        def place(node: Node):
            pos = node.position
            xpos = pos[0] + 0.5 if node.is_bias else pos[0]
            ypos = 0 if node.is_bias else pos[1]
            return (xpos * self.xscale + 0.08, ypos * self.yscale + self.margin)
        return {node: place(node) for node in self.nodes}

    @property
    def weights(self) -> Generator[float, None, None]:
        return (edge.weight for edge in self.edges)

    @property
    def values(self) -> Generator[float, None, None]:
        return (node.value for node in self.nodes)

    @property
    def node_colors(self) -> list[float]:
        return [v for v in self.values]

    @property
    def edge_colors(self) -> list[float]:
        return [w for w in self.weights]

    coolwarm: Colormap = _colormaps.get_cmap('coolwarm'),
    coolwarm = coolwarm[0]
    def draw(self, /, *, label: str="Initial State"):
        """
        Draw the network using matplotlib.
        """
        plt.close()
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_autoscale_on(False)
        #top =self.max_layer_size * 250 + self.margin + 150
        #ax.set_ylim(25, top)
        ax.set_title(label)
        self._draw_nodes(ax)
        self._draw_edges(ax)
        # Label the layers on the graph
        if self.active_layer is not None:
            self._draw_active(ax)

        self._draw_layer_labels(ax)
        if self.expected is not None:
            self._draw_expected(ax)
        plt.show()

    def _draw_nodes(self, ax: Axes):
        """
        Draw the nodes of the network.
        """
        minval = min(*self.values, 0)
        maxval = max(*self.values, 1)
        def draw_nodes(nodelist, /, *,
                       node_size=1000,
                       edgecolors='black',
                       font_color='black',
                       **kwargs):
            draw_networkx_nodes(self.graph, self.positions,
                                nodelist=nodelist,
                                node_size=node_size,
                                node_color=[n.value for n in nodelist],
                                vmin=minval, vmax=maxval,
                                cmap=self.coolwarm,
                                edgecolors=edgecolors,
                                alpha=0.4,
                                ax=ax,
                                **kwargs)
            for node in nodelist:
                pos_x, pos_y = self.positions[node]
                pos = pos_x, pos_y - 0.001
                ax.annotate(node.label, pos,
                            color=font_color,
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
                if node.name is not None:
                    ax.annotate(node.name, (pos_x, pos_y - 0.04),
                                color=font_color,
                                horizontalalignment='center',
                                verticalalignment='center',
                                )
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

    def _draw_edges(self, ax: Axes):
        minweight = min(*self.weights, -0.1)
        maxweight = max(*self.weights, 0.1)
        draw_networkx_edges(self.graph, self.positions,
                            node_size=1000,
                            edge_color=self.edge_colors,
                            edge_cmap=self.coolwarm,
                            edge_vmin=minweight, edge_vmax=maxweight,
                            ax=ax)
        # Label the edges. We'll need to look up node positions.
        positions = self.positions
        # Rotate through some offsets to avoid label collisions
        shifts = (0.065, 0.080, 0.055, 0.075)
        # We group the edges per incomeing node so we can shift the labels
        # to avoid collisions.
        for node in self.graph.nodes:
            for idx, (t, f, edge) in enumerate(self.graph.in_edges(node, data='edge')):
                loc1 = positions[edge.previous]
                loc2 = positions[edge.next]
                loc1_x, loc1_y = loc1
                loc2_x, loc2_y = loc2
                edge_len =  plen(loc1, loc2)
                dx = (loc2_x - loc1_x) / edge_len
                dy = (loc2_y - loc1_y) / edge_len
                # Choose the shift for the label to avoid conflicts
                shift = shifts[idx % len(shifts)]
                loc = loc2_x - dx * shift, loc2_y - dy * shift
                #  Compute the color for the label based on the edge weight
                #  This matches how the edge is colored
                weight = edge.weight
                color = self.coolwarm((weight + 1) / 2)
                ax.annotate(edge.label, loc,
                            color=color,
                            horizontalalignment='center',
                            verticalalignment='center',
                            )

    def _draw_layer_labels(self, ax: Axes):
        """
        Label the layers at the bottom of the graph.
        """
        for layer in self.layers:
            ax.annotate(layer.label, (layer.position * self.xscale + layer_x_offset, layer_y_offset),
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        for layer in self.layers[0:-1]:
            ax.annotate('Bias', ((layer.position + 0.5) * self.xscale + layer_x_offset, layer_y_offset),
                        color='green',
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        for layer in self.layers[1:]:
            ax.annotate(layer.activation.name, (layer.position * self.xscale + layer_x_offset, layer_y2_offset),
                        horizontalalignment='center',
                        verticalalignment='center',
                        )

    def _draw_active(self, ax: Axes):
         """
         Highlight the active layer, if there is one.
         """
         if self.active_layer is not None:
            if self.active_message is not None:
                ax.annotate(self.active_message, (self.active_layer.position * self.xscale + layer_x_offset, layer_y0_offset),
                            color='red',
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
            fancy = FancyBboxPatch((self.active_layer.position * self.xscale + 0.145 * layer_x_offset / 2, layer_y2_offset - 0.02), 0.15, 0.1,
                                    boxstyle='square,pad=0.001',
                                    fc='white', ec='red',
                                    )
            ax.add_patch(fancy)

    def _draw_expected(self, ax: Axes):
        if self.expected is not None:
            # Draw the expected output during a training cycle.
            # expcol is the column for the expected output during training
            expcol = len(self.layers) * self.xscale
            positions = self.positions
            for idx, node in enumerate(self.output_layer.real_nodes):
                pos = positions[node]
                loc = (expcol, pos[1])
                locarrow = (expcol - 0.011, loc[1] - 0.012)
                ax.annotate(f'{self.expected[idx]:.2f}', loc,
                            color='red',
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
                fancy = FancyBboxPatch(locarrow, 0.03, 0.025,
                                        boxstyle='larrow,pad=0.001',
                                        fc='white', ec='red')
                ax.add_patch(fancy)
            loss = self.loss_function(np.array(self.output), np.array(self.expected))
            loss_pos = expcol, layer_y_offset
            ax.annotate(self.loss_function.name, loss_pos, color='red', horizontalalignment='center')
            ax.annotate(f'Loss={loss:.2f}', (loss_pos[0], loss_pos[1] - 0.025), color='red', horizontalalignment='center')

    def show(self, label: str):
        self.draw(label=label)
        return label
    def __call__(self, input: np.array, /, *,
                 epoch: Optional[int] = None
                 ) -> Generator[np.ndarray[Any], Any, np.ndarray[Any]]:
        """
        Evaluate the network for a given input. Returns a generator that produces
        diagrams of the network as it is evaluated. The final value is the output
        from the network as a named tuple.
        """
        layer = self.layers[0]
        layer.values = input
        epoch_label = '' if epoch is None else f'Epoch: {epoch} '
        with self.active(layer, message="Setting input"):
            yield self.show(label=f'{epoch_label}{layer.label}: {''.join(map(repr, layer.values))}')
        for layer in self.layers[1:]:
            with self.active(layer, message=f'Forward: {layer.label}'):
                for node in layer.real_nodes:
                    node.value = sum(edge.weight * edge.previous.value for edge in self.edges)
                    node.value = node.activation(node.value)
                yield self.show(label=f'Forward: {layer.label}')
        yield self.output_type(*self.output_layer.values)

    @contextmanager
    def expecting(self, expected: Sequence[float]):
        """
        Set the expected output for the network during a training pass.
        """
        self.expected = expected
        yield
        self.expected = None

    @contextmanager
    def active(self, layer: Layer, /, *, message: Optional[str] = None):
        """
        Set the active layer for the network during a training pass.
        """
        self.active_layer = layer
        self.active_message = message
        yield
        self.active_layer = None
        self.active_message = None

    def train_one(self, input: np.array, expected: np.array, /, *,
                  epoch:int = 0
                  ) -> Generator[np.ndarray[Any], Any, np.ndarray[Any]]:
        """
        Train the network for a given input and expected output.
        """
        with self.expecting(expected):
            # Forward pass
            yield from self(input, epoch=epoch)
            # Backward pass
            layer = self.layers[-1]
            yield self.show(label=f'Epoch: {epoch} Backward: {layer.label}')
            for layer in reversed(self.layers[0:-1]):
                for node in layer.real_nodes:
                    node.value = sum(edge.weight * edge.next.value for edge in self.edges)
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
    def edges(self) -> Generator[Edge, None, None]:
        return (edge for (f, t, edge) in self.graph.edges(data='edge'))

    @property
    def nodes(self) -> Generator[Node, None, None]:
        return (node for node in self.graph.nodes)

    @property
    def real_nodes(self) -> Generator[Node, None, None]:
        return (node for node in self.graph.nodes if not node.is_bias)

    @property
    def input_layer(self):
        """The input layer of this network."""
        return self.layers[0]

    @property
    def output_layer(self):
        """The output layer of this network."""
        return self.layers[-1]

    @property
    def input(self):
        """The input nodes of this network."""
        return self.input_type(*(n.value for n in self.input_layer.real_nodes))

    @property
    def output(self):
        """The input nodes of this network."""
        return self.output_type(*(n.value for n in self.output_layer.real_nodes))
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
        return f'Network({",".join((str(len(l)) for l in self.layers))}, name={self.name})'

__all__ = ['Network']
