"""
    This file contains the code for the NetGraph class. The NetGraph class is responsible for drawing the network graph.
"""

from colorsys import hsv_to_rgb, rgb_to_hsv
from functools import cached_property
from typing import Generator, Optional, Sequence, Any, cast
import math

from matplotlib import pyplot as plt, colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import FancyBboxPatch

from networkx import DiGraph, draw_networkx_edges, draw_networkx_nodes # type: ignore

from backpropex.node import Input, Node, Output
from backpropex.types import EvalStepResultAny, FloatSeq, StepType, TrainStepResultAny, TrainingData
from backpropex.network import Network

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

class NetGraph:
    """
    Draw a graph of the network.

    This class is responsible for drawing the network graph. It wraps the network
    and exposes the __call__ and train methods of the network, and draws the graph
    after each step.
    """
    net: Network
    margin: float
    xscale: float
    yscale: float

    def __init__(self, net: Network, /, *,
                 margin: float=0.13,
                 ):
        self.net = net
        self.margin = margin
        self.xscale = 1.0 / (len(net.layers) + 0.4)
        self.yscale = 1.0 / (net.max_layer_size + 1)

    @cached_property
    def positions(self):
        """Compute the positions of the nodes in the graph."""
        def place(node: Node):
            pos = node.position
            xpos = pos[0] + 0.5 if node.is_bias else pos[0]
            ypos = 0 if node.is_bias else pos[1]
            return (xpos * self.xscale + 0.08, ypos * self.yscale + self.margin)
        return {node: place(node) for node in self.net.nodes}

    @property
    def node_colors(self) -> list[float]:
        return [v for v in self.net.values]

    @property
    def edge_colors(self) -> list[float]:
        return [w for w in self.net.weights]

    coolwarms: Sequence[Colormap] = colormaps.get_cmap('coolwarm'),
    coolwarm = coolwarms[0]

    def draw(self, /, *, label: str="Initial State"):
        """
        Draw the network using matplotlib.
        """
        plt.close()
        fig, ax = plt.subplots(figsize=(15, 10)) # type: ignore
        ax.set_autoscale_on(False)
        minval = min(*self.net.values, 0)
        maxval = max(*self.net.values, 1)
        minweight = min(*self.net.weights, -0.1)
        maxweight = max(*self.net.weights, 0.1)
        ax.set_title(label)
        self._draw_nodes(ax, minval, maxval)
        self._draw_edges(ax, minweight, maxweight)
        # Label the layers on the graph
        if self.net.active_layer is not None:
            self._draw_active(ax)

        self._draw_layer_labels(ax)
        if self.net.datum_expected is not None:
            self._draw_expected(ax)
        if self.net.epoch_number is not None:
            self._draw_epoch(ax)
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
            draw_networkx_nodes(self.net.graph, self.positions,
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
                if isinstance(node, (Input, Output)):
                    ax.annotate(node.name, (pos_x, pos_y - 0.04), # type: ignore
                                color=font_color,
                                horizontalalignment='center',
                                verticalalignment='center',
                                )
        regular_nodes = [node for node in self.net.nodes if not node.is_bias]
        bias_nodes = [node for node in self.net.nodes if node.is_bias]
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
        for node in self.net.nodes:
            for idx, edge in enumerate(self.net.in_edges(node)):
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
                hsv: tuple[float, float, float] = rgb_to_hsv(*color[0:3])
                hsv = (*hsv[0:2], hsv[2] * 0.9)
                label_color = [*hsv_to_rgb(*hsv), color[3]]
                ax.annotate(edge.label, loc, # type: ignore
                            color=label_color,
                            horizontalalignment='center',
                            verticalalignment='center',
                            )

    def _draw_edges(self, ax: Axes, minweight: float, maxweight: float):
        """
        Draw the edges of the network and their labels.
        """
        edge_colors:Any = self.edge_colors
        draw_networkx_edges(self.net.graph, self.positions,
                            node_size=1000,
                            edge_color=edge_colors,
                            edge_cmap=self.coolwarm,
                            edge_vmin=minweight, edge_vmax=maxweight,
                            ax=ax)
        self._draw_edge_labels(ax, minweight, maxweight)

    def _draw_layer_labels(self, ax: Axes):
        """
        Label the layers at the bottom of the graph.
        """
        for layer in self.net.layers:
            xypos = (layer.position * self.xscale + layer_x_offset, layer_y_offset)
            ax.annotate(layer.label, xypos, # type: ignore
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        for layer in self.net.layers[0:-1]:
            xypos = ((layer.position + 0.5) * self.xscale + layer_x_offset, layer_y_offset)
            ax.annotate('Bias', xypos, # type: ignore
                        color='green',
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        for layer in self.net.layers[1:]:
            xypos = (layer.position * self.xscale + layer_x_offset, layer_y2_offset)
            ax.annotate(layer.activation.name, xypos, # type: ignore
                        horizontalalignment='center',
                        verticalalignment='center',
                        )

    def _draw_active(self, ax: Axes):
         """
         Highlight the active layer, if there is one.
         """
         if self.net.active_layer is not None:
            if self.net.active_message is not None:
                active_msg_pos = (
                        self.net.active_layer.position * self.xscale + layer_x_offset,
                        layer_y0_offset
                )
                ax.annotate(self.active_message, active_msg_pos, # type: ignore
                            color='red',
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
            highlight_pos = (
                    self.net.active_layer.position * self.xscale + 0.145 * layer_x_offset / 2,
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
        if self.net.datum_expected is not None:
            # expcol is the column for the expected output during training
            expcol = len(self.net.layers) * self.xscale
            positions = self.positions
            # Draw the expected output values next to each output node.
            for idx, node in enumerate(self.net.output_layer.real_nodes):
                pos = positions[node]
                loc = (expcol, pos[1])
                locarrow = (expcol - 0.011, loc[1] - 0.012)
                ax.annotate(f'{self.net.datum_expected[idx]:.2f}', loc, # type: ignore
                            color='red',
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
                # Draw a left-facing arrow around the expected output
                fancy = FancyBboxPatch(locarrow, 0.03, 0.025,
                                        boxstyle='larrow,pad=0.001',
                                        fc='white', ec='red')
                ax.add_patch(fancy)
            # If the loss is available, draw it.
            # See the context manager method training_loss.
            if self.net.loss is not None:
                loss_pos = expcol, layer_y_offset
                ax.annotate(self.net.loss_function.name, loss_pos, # type: ignore
                            color='red',
                            horizontalalignment='center')
                ax.annotate(f'Loss={self.net.loss:.2f}', (expcol, layer_y2_offset), # type: ignore
                            color='red',
                            horizontalalignment='center'
                            )

    def _draw_epoch(self, ax: Axes):
        figure = ax.figure
        if figure is None:
            raise ValueError('No figure for axes')
        if self.net.epoch_number is not None:
            if self.net.datum_number is not None:
                figure.text(0.90, 0.88, f'Datum {self.net.datum_number+1}/{self.net.datum_max}', # type: ignore
                            color='red',
                            horizontalalignment='right',
                            verticalalignment='bottom',
                            )
            figure.text(0.125, 0.88, f'Epoch {self.net.epoch_number+1}/{self.net.epoch_max}', # type: ignore
                        color='red',
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        )

    def show(self, label: str):
        self.draw(label=label)
        return label

    def __repr__(self):
        return f'NetGraph({self.net})'

    def __call__(self, input: FloatSeq, /, *,
                 label: Optional[str] = None
                 ) -> Generator[EvalStepResultAny, Any, None]:
        """
        Evaluate the network on the given input and display graphs of the progress.
        """
        extra = f': {label}' if label is not None else ''
        for step in self.net(input, label=label):
            match step.type:
                case StepType.Input:
                    step_label = f'Input: {step.layer.label}{extra}'
                case StepType.Forward:
                    step_label = f'Foward: {step.layer.label}{extra}'
                case StepType.Output:
                    step_label = f'Output: {step.layer.label}{extra}'
            self.draw(label=step_label)
            yield step


    def train(self, data: TrainingData, /, *,
              epochs: int=1000,
              learning_rate: float=0.1
              ) -> Generator[TrainStepResultAny, Any, None]:
        for step in self.net.train(data, epochs=epochs, learning_rate=learning_rate):
            match step.type:
                case StepType.TrainInput:
                    step_label = f'Train Input: {step.layer.label}'
                case StepType.TrainForward:
                    step_label = f'Train Forward: {step.layer.label}'
                case StepType.TrainOutput:
                    step_label = f'Train Output: {step.layer.label}'
                case StepType.TrainLoss:
                    step_label = f'Train Loss: {step.layer.label}'
                case StepType.TrainBackward:
                    step_label = f'Train Backward: {step.layer.label}'
                case StepType.TrainOptimize:
                    step_label = f'Train Optimize: {step.layer.label}'
            self.draw(label=step_label)
            yield step

__all__ = ['NetGraph']
