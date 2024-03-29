"""
    This file contains the code for the NetGraph class. The NetGraph class is responsible for drawing the network graph.
"""

from collections.abc import Generator, Sequence
from typing import Optional, Any, cast, overload
import math

from colorsys import hsv_to_rgb, rgb_to_hsv
from functools import cached_property

from matplotlib import pyplot as plt, colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import FancyBboxPatch

from networkx import (
     DiGraph,
     draw_networkx_edges, # type: ignore
     draw_networkx_nodes, # type: ignore
 )
from backpropex.layer import Layer

from backpropex.node import Input, Node, Output
from backpropex.types import (
    FloatSeq, TrainingData
)
from backpropex.steps import (
    LayerStepResult,
    StepResult,
    StepType,
    TrainLossStepResult, TrainStepResult,
    StepResultAny, EvalStepResultAny, TrainStepResultAny,
)
from backpropex.protocols import (
    EvalProtocol, Filter, GraphProtocol, Trace, TrainProtocol, NetProtocol,
    LossFunction,
)
from backpropex.utils import make

# Constants for drawing the network
layer_x_offset = 0.022
layer_y_offset = 0.05 # Main row
layer_y2_offset = layer_y_offset - 0.025 # second row, below main row
layer_y0_offset = layer_y_offset + 0.025 # zeroth row, above main row
layer_label_x_offset = 0.4


def plen(p: tuple[float, float], n: tuple[float, float]) -> float:
    """{Pythagorhean length of a line. expressed as two points"""
    x2 = (p[0] - n[0]) ** 2.0
    y2 = (p[1] - n[1]) ** 2.0
    return math.sqrt(x2 + y2)

class NetGraph(EvalProtocol, TrainProtocol, GraphProtocol):
    """
    Draw a graph of the network.

    This class is responsible for drawing the network graph. It wraps the network
    and exposes the __call__ and train methods of the network, and draws the graph
    after each step.
    """
    trainer: Optional[TrainProtocol]
    margin: float
    xscale: float
    yscale: float

    _filter: Optional[Filter|type[Filter]] = None
    _trace: Optional[Trace|type[Trace]] = None

    graph: DiGraph

    title: str

    loss_function: LossFunction


    @overload
    def __init__(self, net: NetProtocol, /, *,
                 margin: float=0.13,
                 filter: Optional[Filter|type[Filter]] = None,
                 trace: Optional[Trace|type[Trace]] = None,
                 title: Optional[str] = None,
                 ) -> None:
        ...
    @overload
    def __init__(self, trainer: TrainProtocol, /, *,
                 margin: float=0.13,
                 filter: Optional[Filter|type[Filter]] = None,
                 trace: Optional[Trace|type[Trace]] = None,
                 title: Optional[str] = None,
                 ) -> None:
        ...

    def __init__(self, proxy: NetProtocol|TrainProtocol, /, *,
                 margin: float=0.13,
                 filter: Optional[Filter|type[Filter]] = None,
                 trace: Optional[Trace|type[Trace]] = None,
                 title: Optional[str] = None,
                 ) -> None:
        """
        Initialize the graph drawer for either  a network or a trainer.
        """
        if isinstance(proxy, NetProtocol):
            self.net = proxy
            self.trainer = None
        else:
            self.net = proxy.net
            self.trainer = proxy
            self.loss_function = proxy.loss_function
        self.margin = margin
        self.xscale = 1.0 / (len(self.net.layers) + 0.4)
        self.yscale = 1.0 / (self.net.max_layer_size + 1)

        self.graph = DiGraph()

        self.title = title or self.net.name or 'Network'

        if filter is not None:
            self._filter = make(filter, Filter)
        if trace is not None:
            self._trace = make(trace, Trace)

        coolwarms: Sequence[Colormap] = colormaps.get_cmap('coolwarm'),
        self.color_map = coolwarms[0]

        for layer in self.net.layers:
            for node in layer.nodes:
                self.graph.add_node(node) # type: ignore
                for edge in node.edges_to:
                    prev, next = edge.from_, edge.to_
                    self.graph.add_edge(prev, next, edge=edge) # type: ignore

    @cached_property
    def positions(self):
        """Compute the positions of the nodes in the graph."""
        def place(node: Node):
            pos = node.position
            xpos = pos[0] + 0.5 if node.is_bias else pos[0]
            ypos = 0 if node.is_bias else pos[1]
            return (xpos * self.xscale + 0.08, ypos * self.yscale + self.margin)
        return {
            node: place(node)
            for node in self.net.nodes
        }

    @property
    def node_colors(self) -> list[float]:
        return [v for v in self.net.values]

    @property
    def edge_colors(self) -> list[float]:
        return [w for w in self.net.weights]

    def draw(self, result: Optional[StepResultAny] = None, /, *, label: Optional[str]=None):
        """
        Draw the network using matplotlib.
        """
        plt.close()
        fig, ax = plt.subplots(figsize=(15, 10)) # type: ignore
        ax.set_autoscale_on(False)
        minval = min(*self.net.values, 0.1)
        maxval = max(*self.net.values, 1.1)
        minweight = min(*self.net.weights, -0.1, 0.1)
        maxweight = max(*self.net.weights, 0.1, 0.1)
        norm = Normalize(vmin=minval, vmax=maxval)
        wnorm = Normalize(vmin=minweight, vmax=maxweight)
        mappable = ScalarMappable(
                         cmap=self.color_map,
                         norm=norm,
                     )
        wmappable = ScalarMappable(
                         cmap=self.color_map,
                         norm=wnorm,
                     )
        ax.set_title(label or self.title)
        # Label the layers on the graph
        if self.net.active_layer is not None:
            self._draw_active(ax)
        if result is not None:
            if isinstance(result, TrainStepResult):
                tresult = cast(TrainStepResultAny, result)
                self._draw_expected(ax, tresult)
                self._draw_epoch(ax, tresult)
            self._draw_step(ax, result)

        self._draw_nodes(ax, mappable, result)
        self._draw_edges(ax, wmappable, result)

        self._draw_layer_labels(ax)
        cax1 = fig.add_axes((0.905, 0.50, 0.007, 0.38)) # type: ignore
        cax2 = fig.add_axes((0.905, 0.11, 0.006, 0.38)) # type: ignore
        fig.colorbar(drawedges=False, # type: ignore
                     cax=cax1,
                     norm=norm,
                     extend='both',
                     label='Node value',
                     mappable=mappable,
                     )
        fig.colorbar(drawedges=False, # type: ignore
                     cax=cax2,
                     norm=wnorm,
                     extend='both',
                     label='Edge weight',
                     mappable=wmappable,
                     )
        plt.show() # type: ignore

    def _draw_nodes(self, ax: Axes, mappable: ScalarMappable, step: StepResultAny|None):
        """
        Draw the nodes of the network.
        """
        def draw_nodes(nodelist: list[Node], /, *,
                       node_size: int=1000,
                       edgecolors: str|float|int ='black',
                       font_color: str|float|int ='black',
                       **kwargs: Any):
            node_colors = [
                cast(tuple[float, float, float, float], mappable.to_rgba(value, alpha=0.4)) # type: ignore
                for value in (n.value for n in nodelist)
                ]
            draw_networkx_nodes(
                self.graph, self.positions,
                    nodelist=nodelist,
                    node_size=node_size,
                    node_color=node_colors, # type: ignore
                    edgecolors=edgecolors,
                    ax=ax,
                    **kwargs
                )
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

    def _draw_edge_labels(self, ax: Axes, mappable: ScalarMappable, step: StepResultAny|None):
        """Label the edges of the network."""
        # Label the edges. We'll need to look up node positions.
        positions = self.positions
        # Rotate through some offsets to avoid label collisions
        shifts = (0.065, 0.080, 0.055, 0.075)
        # We group the edges per incoming node so we can shift the labels
        # to avoid collisions.
        for node in self.net.nodes:
            for idx, edge in enumerate(node.edges_to):
                loc1 = positions[edge.from_]
                loc2 = positions[edge.to_]
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
                color = mappable.to_rgba(weight) # type: ignore
                hsv: tuple[float, float, float] = rgb_to_hsv(*color[0:3])
                hsv = (*hsv[0:2], hsv[2] * 0.9)
                label_color = [*hsv_to_rgb(*hsv), color[3]]
                layer = edge.to_.layer
                label = edge.label
                if step is not None:
                    match step.type:
                        case StepType.TrainBackward:
                            llayer: Layer = cast(LayerStepResult[Any], step).layer
                            if layer.idx >= llayer.idx:
                                partial = llayer.gradient[edge.to_.idx]
                                label = f'{label}\n{partial:.2f}'
                        case StepType.TrainOptimize:
                            llayer: Layer = cast(LayerStepResult[Any], step).layer
                            if layer.idx == llayer.idx:
                                delta = llayer.weight_delta[edge.to_.idx]
                                # Show the change just on the current layer
                                label = f'{edge.weight-delta:.2f}\n{delta:.2f}'
                        case _:
                            pass
                ax.annotate(label, loc, # type: ignore
                            color=label_color,
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
    def _draw_edges(self, ax: Axes, mappable: ScalarMappable, step: StepResultAny|None):
        """
        Draw the edges of the network and their labels.
        """
        edge_color_values:Any = self.edge_colors
        edge_colors = [
            tuple[float](mappable.to_rgba(w)) # type: ignore
            for w in edge_color_values
        ]
        draw_networkx_edges(self.graph, self.positions,
                            node_size=1000,
                            edge_color=edge_colors, # type: ignore
                            ax=ax)
        self._draw_edge_labels(ax, mappable, step)

    def _draw_layer_labels(self, ax: Axes):
        """
        Label the layers at the bottom of the graph.
        """
        def row_start(layer: Layer, offset: float = 0.0) -> float:
            return (layer.idx + layer_label_x_offset + offset) * self.xscale + layer_x_offset
        for layer in self.net.layers:
            xypos = (
                row_start(layer),
                layer_y_offset)
            ax.annotate(layer.label, xypos, # type: ignore
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        for layer in self.net.layers[0:-1]:
            xypos = (row_start(layer, 0.5), layer_y_offset)
            ax.annotate('Bias', xypos, # type: ignore
                        color='green',
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        for layer in self.net.layers[1:]:
            xypos = (row_start(layer), layer_y2_offset)
            ax.annotate(layer.activation_fn.name, xypos, # type: ignore
                        horizontalalignment='center',
                        verticalalignment='center',
                        )

    def draw_highlight(self, ax: Axes, expcol: float):
        if self.net.active_message is not None:
            ax.annotate(self.active_message, (expcol, layer_y0_offset), # type: ignore
                        color='red',
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        width = 0.64 / (len(self.net.layers) + 1)
        highlight_pos = (
            expcol - layer_x_offset / 2 - 0.025 + width / 2,
            layer_y2_offset - 0.02
        )
        fancy = FancyBboxPatch(highlight_pos, width, 0.98,
                                boxstyle='square,pad=0.001',
                                fc=(0.9, 1.0, 0.92), ec=(0.3, 1.0, 0.3),
                                )
        ax.add_patch(fancy)

    def _draw_active(self, ax: Axes):
         """
         Highlight the active layer, if there is one.
         """
         if self.net.active_layer is not None:
            expcol = self.net.active_layer.idx * self.xscale + layer_x_offset
            self.draw_highlight(ax, expcol)

    def _draw_expected(self, ax: Axes, result: TrainStepResultAny):
        """
        Draw the expected output column during a training cycle.

        This is only done if the expected output has been made
        available via the context manager method `expected_output()`.
        """
        # expcol is the column for the expected output during training
        expcol = len(self.net.layers) * self.xscale
        positions = self.positions
        # Draw the expected output values next to each output node.
        # If the loss is available, draw it.
        # See the context manager method training_loss.
        if isinstance(result, TrainLossStepResult):
            ax.annotate(f'Loss={result.loss:.2f}', (expcol, layer_y2_offset), # type: ignore
                        color='red',
                        horizontalalignment='center',
                        verticalalignment='center',
                    )
            self.draw_highlight(ax, expcol)
        for idx, node in enumerate(self.net.output_layer.real_nodes):
            pos = positions[node]
            loc = (expcol, pos[1])
            locarrow = (expcol - 0.011, loc[1] - 0.012)
            ax.annotate(f'{self.trainer.datum.expected[idx]:.2f}', loc, # type: ignore
                        color='red',
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
            # Draw a left-facing arrow around the expected output
            fancy = FancyBboxPatch(locarrow, 0.03, 0.025,
                                    boxstyle='larrow,pad=0.001',
                                    fc='white', ec='red')
            ax.add_patch(fancy)
        ax.annotate('Expected', (expcol, layer_y0_offset), # type: ignore
                    color='red',
                    horizontalalignment='center',
                    verticalalignment='center',
                )
        ax.annotate(self.trainer.loss_function.name, (expcol, layer_y_offset), # type: ignore
                    color='red',
                    horizontalalignment='center',
                    verticalalignment='center',
                )

    def _draw_epoch(self, ax: Axes, result: TrainStepResultAny):
        figure = ax.figure
        if figure is None:
            raise ValueError('No figure for axes')
        msg = (f'Epoch {result.epoch+1}/{result.epoch_max}'   # type: ignore
             + f' Datum {result.datum_no+1}/{result.datum_max}') # type: ignore
        figure.text(0.125, 0.88, msg, # type: ignore
                    color='red',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    )

    def _draw_step(self, ax: Axes, result: StepResultAny):
        figure = ax.figure
        if figure is None:
            raise ValueError('No figure for axes')
        step = cast(StepResult[StepType], result)
        msg = f'Step {step.type}'
        figure.text(0.90, 0.88, msg, # type: ignore
                    color='red',
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    )

    @overload
    def __call__(self, input: FloatSeq, /, *,
                label: Optional[str] = None,
                filter: Optional[Filter|type[Filter]] = None,
                trace: Optional[Trace|type[Trace]] = None,
                **kwargs: Any,
                ) -> Generator[EvalStepResultAny, Any, None]:
        ...
    @overload
    def __call__(self, data: TrainingData, /, *,
                 epochs: int=1000,
                 learning_rate: float=0.1,
                 filter: Optional[Filter|type[Filter]] = None,
                 trace: Optional[Trace|type[Trace]] = None,
                 **kwargs: Any,
            ) -> Generator[TrainStepResultAny, Any, None]:
        ...
    def __call__(self, data: FloatSeq|TrainingData, /, *,
                 epochs: int=1000,
                 learning_rate: float=0.1,
                 label: Optional[str] = None,
                 filter: Optional[Filter|type[Filter]] = None,
                 trace: Optional[Trace|type[Trace]] = None,
                 **kwargs: Any,
            ) -> Generator[StepResultAny, Any, None]:
        """
        Evaluate or train the network. The network is drawn after each step (unless
        disabled by a filter).

        :param data: The input data to evaluate or train on.
        :param epochs: The number of epochs to train for. Only applies if applied to a `Trainer`.
        :param learning_rate: The learning rate to use. Only applies if applied to a `Trainer`.
        :param label: The label to use for the diagram.
        :param filter: A filter to apply to the steps, or None to accept all steps. Only steps
                       accepted by the filter will be drawn.
        :param trace: a `Trace` to apply to the steps. The trace will be called for each step
                      and can print or collect the steps for later examination.
        """
        if trace:
            _trace = make(trace, Trace)
        else:
            _trace = None
        def do_trace[R: StepResultAny](result: R) -> R:
            if self._trace is not None:
                self._trace(result.type, result)
            if _trace is not None:
                _trace(result.type, result)
            return result
        if filter is not None:
            filter = make(filter, Filter)
        if self.trainer is None:
            results = self.eval(cast(FloatSeq, data),
                                label=label,
                                filter=filter,
                                **kwargs
                                )
        else:
            results = self.train(cast(TrainingData, data),
                                epochs=epochs,
                                filter=filter,
                                learning_rate=learning_rate,
                                **kwargs
                                )
        yield from (
            do_trace(step)
            for step in results
        )

    def eval(self, data: FloatSeq, /, *,
                label: Optional[str] = None,
                filter: Optional[Filter] = None,
                **kwargs: Any,
            ) -> Generator[EvalStepResultAny, Any, None]:
        """
        Evaluate the network for a given input. Returns a generator that produces
        diagrams of the network as it is evaluated. The final value is the output
        from the network as a named tuple.
        """
        for step in self.net(data):
            if (not filter or filter(step.type, step)):
                if (not self._filter or self._filter(step.type, step)):
                    self.draw(step, label=label)
            yield step

    def train(self, data: TrainingData, /, *,
                epochs: int=1000,
                learning_rate: float=0.1,
                filter: Optional[Filter] = None,
                **kwargs: Any,
                ) -> Generator[TrainStepResultAny, Any, None]:
        """
        Train the network on the given training data.
        """
        if self.trainer is None:
            raise ValueError('Cannot train a network without a trainer')
        for step in self.trainer(data, epochs=epochs, learning_rate=learning_rate):
            if (not filter or filter(step.type, step)):
                if (not self._filter or self._filter(step.type, step)):
                    self.draw(step)
            yield step

    def __repr__(self):
        return f'NetGraph({self.net})'

__all__ = ['NetGraph']
