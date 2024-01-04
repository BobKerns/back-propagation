"""
 A neural network. This module contains the `Network` class, which represents
    a neural network. The network is represented as a directed graph, with the
    nodes representing the neurons and the edges representing the connections
    between the neurons.

    The network is drawn using matplotlib, and the network can be evaluated
    for a given input, and the result of the evaluation is returned as a
    named tuple.

    The network can be trained using backpropagation. The network is trained
    using a set of training data, and the network is updated using the
    gradient descent algorithm.

    The network can be drawn during training, and the progress of the training
    can be monitored.

    This is not a general-purpose neural network library. It is intended to
    be used as a teaching tool, to demonstrate how neural networks work.
    It is not optimized for speed, and it is not optimized for memory usage.
    Very little vectorization is used, so it will appear very different from
    other examples you may have seen.
"""

from contextlib import contextmanager
from functools import cached_property
from typing import Any, Generator, NamedTuple, Optional, Protocol, Sequence, cast
import re
import math
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize, rgb_to_hsv, hsv_to_rgb # type: ignore
from matplotlib.patches import FancyBboxPatch

from networkx import DiGraph, draw_networkx_edges, draw_networkx_nodes  # type: ignore
import numpy as np

from backpropex.types import LayerType, NPFloats, FloatSeq
from backpropex.activation import ACT_ReLU, ACT_Sigmoid, ActivationFunction
from backpropex.edge import Edge
from backpropex.layer import Layer
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

class NetTuple(Protocol):
    """Constructor for input or output named tuples"""
    def __call__(self, *args: float) -> tuple[float, ...]:
        ...
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
    input_type: NetTuple
    output_type: NetTuple
    xscale: float
    yscale: float

    # Progress information for drawing the network
    # The layer that is currently being evaluated
    active_layer: Optional[Layer] = None
    active_message: Optional[str] = None
    datum_value: Optional[NPFloats] = None
    datum_expected: Optional[NPFloats] = None

    def __init__(self, *layers: int,
                 name: Optional[str] = None,
                 loss_function: LossFunction=MeanSquaredError,
                 activations: Optional[Sequence[ActivationFunction]]=None,
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
        def sanitize(name: str) -> str:
            return '_'.join((s for s in re.split(r'[^a-zA-Z0-9_]+', name) if s != ''))
        fields = [
            (sanitize(node.name), float)
            for node
            in self.input_layer.real_nodes
            if node.name is not None
        ]
        self.input_type = NamedTuple(f'{sanitize(self.name)}_input', fields)
        fields = [
            (sanitize(node.name), float)
            for node
            in self.output_layer.real_nodes
            if node.name is not None
        ]
        self.output_type = NamedTuple(f'{sanitize(self.name)}_output', fields)
    def layer_type(self, idx: int, nlayers: int = -1):
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
                self.graph.add_edge(node, next_node, edge=edge) # type: ignore

    @property
    def labels(self) -> dict[Node, str]:
        return {n: n.label for n in self.nodes}

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

    coolwarms: Sequence[Colormap] = colormaps.get_cmap('coolwarm'),
    coolwarm = coolwarms[0]
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
        norm = Normalize(vmin=minval, vmax=maxval)
        cax1 = fig.add_axes((0.905, 0.50, 0.007, 0.38)) # type: ignore
        cax2 = fig.add_axes((0.905, 0.11, 0.006, 0.38)) # type: ignore
        fig.colorbar(drawedges=False, # type: ignore
                     cax=cax1,
                     norm=norm,
                     extend='both',
                     label='Node value',
                     mappable=ScalarMappable(
                         cmap=self.coolwarm,
                         norm=norm,
                     ))
        wnorm = Normalize(vmin=minweight, vmax=maxweight)
        fig.colorbar(drawedges=False, # type: ignore
                     cax=cax2,
                     norm=wnorm,
                     extend='both',
                     label='Edge weight',
                     mappable=ScalarMappable(
                         cmap=self.coolwarm,
                         norm=wnorm,
                     ))
        plt.show() # type: ignore

    def _draw_nodes(self, ax: Axes, minval: float, maxval: float):
        """
        Draw the nodes of the network.
        """
        def draw_nodes(nodelist: list[Node], /, *,
                       node_size: int=1000,
                       edgecolors: str|float|int ='black',
                       font_color: str|float|int ='black',
                       **kwargs: Any):
            draw_networkx_nodes(self.graph, self.positions,
                                nodelist=nodelist,
                                node_size=node_size,
                                node_color=cast(str, [n.value for n in nodelist]),
                                vmin=minval, vmax=maxval,
                                cmap=self.coolwarm,
                                edgecolors=edgecolors,
                                alpha=0.4,
                                ax=ax,
                                **kwargs)
            for node in nodelist:
                pos_x, pos_y = self.positions[node]
                pos = pos_x, pos_y - 0.001
                ax.annotate(node.label, pos, # type: ignore
                            color=font_color,
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
                if not node.is_bias and node.name is not None:
                    ax.annotate(node.name, (pos_x, pos_y - 0.04), # type: ignore
                                color=font_color,
                                horizontalalignment='center',
                                verticalalignment='center',
                                )
        regular_nodes = [node for node in self.nodes if not node.is_bias]
        bias_nodes = [node for node in self.nodes if node.is_bias]
        # Draw the regular nodes first
        draw_nodes(regular_nodes)
        # Draw the bias nodes distinctively differently.
        draw_nodes(bias_nodes,
                   node_shape='s',
                   node_size=500,
                   linewidths=2.0,
                   edgecolors='green',
                   font_color='green')

    def _draw_edge_labels(self, ax: Axes, minweight: float, maxweight: float):
        """Label the edges of the network."""
        # Label the edges. We'll need to look up node positions.
        positions = self.positions
        # Rotate through some offsets to avoid label collisions
        shifts = (0.065, 0.080, 0.055, 0.075)
        # We group the edges per incomeing node so we can shift the labels
        # to avoid collisions.
        norm = Normalize(vmin=minweight, vmax=maxweight)
        for node in self.nodes:
            for idx, edge in enumerate(self.in_edges(node)):
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
                # Darken the labels a bit for readability.  Any more than this
                # makes the mid-tones too gray.
                color = self.coolwarm(norm(weight))
                hsv: NPFloats = rgb_to_hsv(color[0:3])
                hsv[2] = hsv[2] * 0.9
                label_color = hsv_to_rgb(hsv).tolist() + [color[3]]
                ax.annotate(edge.label, loc, # type: ignore
                            color=label_color,
                            horizontalalignment='center',
                            verticalalignment='center',
                            )

    def _draw_layer_labels(self, ax: Axes):
        """
        Label the layers at the bottom of the graph.
        """
        for layer in self.layers:
            xypos = (layer.position * self.xscale + layer_x_offset, layer_y_offset)
            ax.annotate(layer.label, xypos, # type: ignore
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        for layer in self.layers[0:-1]:
            xypos = ((layer.position + 0.5) * self.xscale + layer_x_offset, layer_y_offset)
            ax.annotate('Bias', xypos, # type: ignore
                        color='green',
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        for layer in self.layers[1:]:
            xypos = (layer.position * self.xscale + layer_x_offset, layer_y2_offset)
            ax.annotate(layer.activation.name, xypos, # type: ignore
                        horizontalalignment='center',
                        verticalalignment='center',
                        )

    def _draw_active(self, ax: Axes):
         """
         Highlight the active layer, if there is one.
         """
         if self.active_layer is not None:
            if self.active_message is not None:
                active_msg_pos = (
                        self.active_layer.position * self.xscale + layer_x_offset,
                        layer_y0_offset
                )
                ax.annotate(self.active_message, active_msg_pos, # type: ignore
                            color='red',
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
            highlight_pos = (
                    self.active_layer.position * self.xscale + 0.145 * layer_x_offset / 2,
                    layer_y2_offset - 0.02
                )
            fancy = FancyBboxPatch(highlight_pos, 0.15, 0.1,
                                    boxstyle='square,pad=0.001',
                                    fc='pink', ec='red',
                                    )
            ax.add_patch(fancy)

    def _draw_expected(self, ax: Axes):
        """
        Draw the expected output column during a training cycle.

        This is only done if the expected output has been made
        available via the context manager method `expected_output()`.
        """
        if self.datum_expected is not None:
            # expcol is the column for the expected output during training
            expcol = len(self.layers) * self.xscale
            positions = self.positions
            # Draw the expected output values next to each output node.
            for idx, node in enumerate(self.output_layer.real_nodes):
                pos = positions[node]
                loc = (expcol, pos[1])
                locarrow = (expcol - 0.011, loc[1] - 0.012)
                ax.annotate(f'{self.datum_expected[idx]:.2f}', loc, # type: ignore
                            color='red',
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
                # Draw a left-facing arrow around the expected output
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
    def __call__(self, input: FloatSeq, /, *,
                 label: Optional[str] = None
                 ) -> Generator[Any, Any, None]:
        """
        Evaluate the network for a given input. Returns a generator that produces
        diagrams of the network as it is evaluated. The final value is the output
        from the network as a named tuple.
        """
        layer = self.layers[0]
        layer.values = input
        extra = f': {label}' if label is not None else ''
        with self.step_active(layer, message="Setting input"):
            yield self.show(label=f'{layer.label}{extra}')
        for layer in self.layers[1:]:
            with self.step_active(layer, message=f'Forward{extra}'):
                for node in layer.real_nodes:
                    value = sum(edge.weight * edge.previous.value for edge in self.edges)
                    node.value = node.activation(value)
                yield self.show(label=f'Forward: {layer.label}{extra}')
        # Yeld the result back to the caller.
        # We need a better protocol for this.
        yield self.output_type(*self.output_layer.values)

    @contextmanager
    def step_active(self, layer: Layer, /, *, message: Optional[str] = None):
        """
        Set the active layer for the network during a training pass.
        """
        self.active_layer = layer
        self.active_message = message
        yield layer
        self.active_layer = None
        self.active_message = None


    def train_one(self, input: NPFloats, expected: NPFloats, /,
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
                for idx, (input, expected) in enumerate(data):
                    input = np.array(input)
                    expected = np.array(expected)

    @property
    def edges(self) -> Generator[Edge, None, None]:
        return (
            cast(Edge, edge)
            for (f, t, edge) # type: ignore
            in self.graph.edges(data=cast(bool, 'edge'))
            )

    @property
    def nodes(self) -> Generator[Node, None, None]:
        return (cast(Node, node) for node in self.graph.nodes) # type: ignore

    @property
    def real_nodes(self) -> Generator[Node, None, None]:
        return (node for node in self.nodes if not node.is_bias)

    def in_edges(self, node: Node) -> Generator[Edge, None, None]:
        return (
            cast(Edge, edge)
            for (f, t, edge) # type: ignore
            in self.graph.in_edges(node, data=cast(bool, 'edge'))
            )

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
