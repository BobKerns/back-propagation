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

from typing import Any, Callable, Generator, NamedTuple, Optional
from collections.abc import Iterable
from contextlib import contextmanager

import numpy as np

from backpropex.builder import DefaultBuilder, sanitize
from backpropex.filters import FilterChain
from backpropex.randomizer import HeEtAl

from backpropex.steps import (
    EvalOutputStepResult,
    EvalStepResultAny,
    InitStepResult,
    StepResultAny,
    StepType,
    EvalForwardStepResult, EvalInputStepResult,
)
from backpropex.types import (
    FloatSeq, LayerType, NetTuple,
)
from backpropex.protocols import Builder, BuilderContext, NetProtocol, Filter, Randomizer
from backpropex.edge import Edge
from backpropex.layer import Layer
from backpropex.node import Node
from backpropex.filters import FilterChain

def _ids():
    """Generate unique ids."""
    idx = 0
    while True:
        yield idx
        idx += 1

ids = _ids()

class Network(NetProtocol):
    """
    A neural network.
    """
    #graph: DiGraph = DiGraph()
    input_type: NetTuple
    output_type: NetTuple
    layers: list[Layer]
    max_layer_size: int
    name: str

    _filter: Optional[Filter] = None

    def __init__(self, *layers: int,
                 name: Optional[str] = None,
                 builder: Optional[Builder|type[Builder]] = DefaultBuilder,
                 randomizer: Optional[Randomizer|type[Randomizer]] = HeEtAl,
                 filter: Optional[Filter|type[Filter]] = None,
                 **kwargs: Any
                 ):
        """
        A neural network.

        :param layers: The number of nodes in each layer.
        :param loss_function: The loss function for this network.
        :param activation_functions: The activation function for each layer.
        """
        self.net = self
        self.layers = list()

        if isinstance(filter, type):
            self._filter = filter()
        else:
            self._filter = filter

        self.name = name if name is not None else f'Network_{next(ids)}'

        context = self.Context(self)

        if isinstance(builder, type):
            builder()(context, *layers, **kwargs)
        elif isinstance(builder, Builder):
            builder(context, *layers, **kwargs)
        else:
            raise TypeError(f'Invalid builder type: {builder}')

        if isinstance(randomizer, type):
            self.randomizer = randomizer()
        elif isinstance(randomizer, Randomizer):
            self.randomizer = randomizer
        else:
            raise TypeError(f'randomizer must be a Randomizer or a Randomizer type, not {type(randomizer)}')

        # Set weights
        for (prev_layer, next_layer) in zip(self.layers[:-1], self.layers[1:]):
            prevCount = len(prev_layer.nodes)
            if next_layer.layer_type == LayerType.Output:
                nextCount = len(next_layer.nodes)
            else:
                nextCount = len(next_layer.nodes)
            w = self.randomizer((prevCount, nextCount))
            for (nidx, node) in enumerate(next_layer.nodes):
                for (pidx, edge) in enumerate(node.edges_to):
                    edge.weight = w[pidx][nidx]

        self.max_layer_size = max(len(layer) for layer in self.layers)
        self.input_type = self.mk_namedtuple('input', self.input_layer)
        self.output_type = self.mk_namedtuple('output', self.output_layer)

    @contextmanager
    def filterCheck[R: StepResultAny](
        self, type: StepType,
        mk_result: Callable[[], R],
    /) -> Generator[R|None, Any, None]:
        """
        Check if the filter allows the step to proceed.
        """
        if self._filter is not None:
            if self._filter(type, None):
                result = mk_result()
                if result.type != type:
                    raise ValueError(f'Wrong result type: {result.type}')
                if self._filter(type, result):
                    yield result
                else:
                    yield None
            else:
                yield None
            if self._filter(type, False):
                raise StopIteration
        else:
            result = mk_result()
            if result.type != type:
                raise ValueError(f'Wrong result type: {result.type}')
            yield result

    def mk_namedtuple(self, suffix: str, layer: Layer) -> NetTuple:
        fields = [
            (sanitize(node.name), float)
            for node
            in layer.real_nodes
            if node.name is not None
        ]
        return NamedTuple(f'{sanitize(self.name)}_{suffix}', fields)
    @contextmanager
    def step_active(self, layer: Layer, /):
        """
        Set the active layer for the network during a training pass.
        """
        self.active_layer = layer
        yield layer
        self.active_layer = None
        self.active_message = None

    @contextmanager
    def filter(self, _filter: Filter|type[Filter]|None, /) -> Generator[Filter |  None, Any, None]:
        """
        Emable a filter.
        """
        if _filter is None:
            yield self._filter
        else:
            if isinstance(_filter, type):
                _ifilter = _filter()
            else:
                _ifilter = _filter
            prev = self._filter
            if prev is not None:
                _ifilter = FilterChain(prev, _ifilter)
            self._filter = _ifilter
            yield _ifilter
            self._filter = prev

    def __call__(self, input: FloatSeq, /, *,
                 label: Optional[str] = None,
                 filter: Optional[Filter|type[Filter]] = None,
                 ) -> Generator[EvalStepResultAny, Any, None]:
        """
        Evaluate the network for a given input. Returns a generator that produces
        diagrams of the network as it is evaluated. The final value is the output
        from the network as a named tuple.
        """
        with self.filter(filter):
            with self.filter(self._filter):
                with self.filterCheck(StepType.Initialized,
                                lambda : InitStepResult(StepType.Initialized)) as step:
                    if step:
                        yield step
                layer = self.layers[0]
                layer.values = input
                with self.step_active(layer):
                    in_tuple = self.input_type(*input)
                    with self.filterCheck(StepType.Input,
                                    lambda : EvalInputStepResult(StepType.Input, layer=layer, input=in_tuple)) as step:
                        if step:
                            yield step
                for layer in self.layers[1:]:
                    with self.step_active(layer):
                        for node in layer.real_nodes:
                            value = sum(
                                (
                                    edge.weight * edge.from_.value
                                    for edge in node.edges_to
                                    ))
                            node.value = node.activation(value)
                        def mk_forward():
                            return EvalForwardStepResult(StepType.Forward,
                                                        layer=layer,
                                                        values=tuple(node.value for node in layer.real_nodes))
                        def mk_output():
                            return EvalOutputStepResult(StepType.Output,
                                                        layer=self.output_layer,
                                                        output=self.output_type(*self.output))
                        def mk_step[R: StepResultAny]() :
                            match(layer.layer_type):
                                case LayerType.Output:
                                    return StepType.Output, mk_output
                                case _:
                                    return StepType.Forward, mk_forward
                        with self.filterCheck(*mk_step()) as step:
                            if step:
                                yield step

    # For use by the builder
    class Context(BuilderContext):
        """
        The context for the builder.
        """
        net: NetProtocol
        def __init__(self, network: 'Network', /):
            self.net = network
        def add_layer(self, layer: Layer, /):
            """
            Add a layer to the network.
            """
            self.net.layers.append(layer)

        def add_layers(self, layers: Iterable[Layer], /):
            """
            Add layers to the network.
            """
            for layer in layers:
                self.add_layer(layer)

        @property
        def layers(self) -> list[Layer]:
            return self.net.layers

        @property
        def nodes(self) -> Generator[Node, None, None]:
            """
            The nodes in the network.
            """
            return self.net.nodes


        @property
        def real_nodes(self) -> Generator[Node, None, None]:
            """
            The nodes in the network, excluding bias nodes.
            """
            return self.net.real_nodes

        @property
        def edges(self) -> Generator[Edge, None, None]:
            """
            The edges in the network.
            """
            return self.net.edges

        def add_edge(self, edge: Edge, /):
            """
            Add an edge to the network.
            """
            from_ = edge.from_
            to_ = edge.to_
            from_.addFrom(edge)
            to_.addTo(edge)    # type: ignore
        def add_edges(self, edges: Iterable[Edge], /):
            """
            Add edges to the network.
            """
            for edge in edges:
                self.add_edge(edge) # type: ignore

    @property
    def edges(self) -> Generator[Edge, None, None]:
        return (
            edge
            for layer in self.net.layers
            for node in layer.nodes
            for edge in node.edges
            )

    @property
    def nodes(self) -> Generator[Node, None, None]:
        return (
            node
            for layer in self.net.layers
            for node in layer.nodes
            )

    @property
    def real_nodes(self) -> Generator[Node, None, None]:
        return (node for node in self.nodes if not node.is_bias)

    @property
    def weights(self) -> Generator[float, None, None]:
        return (edge.weight for edge in self.edges)

    @property
    def values(self) -> Generator[float, None, None]:
        return (node.value for node in self.nodes)

    @property
    def labels(self) -> dict[Node, str]:
        return {
            n:n.label
            for n in self.real_nodes
        }

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
        """The output values of this network as a namedtuple."""
        return self.output_type(*(n.value for n in self.output_layer.real_nodes))

    @property
    def output_array(self):
        """The output values of this network as an array."""
        return np.array([n.value for n in self.output_layer.real_nodes])
    @property
    def hidden_layers(self) -> tuple[Layer, ...]:
        """The hidden layers of this network."""
        return tuple(self.layers[1:-1])

    def __getitem__(self, idx: int):
        """Get a layer by index."""
        return self.layers[idx]

    def __len__(self):
        """The number of layers in this network."""
        return len(self.layers)

    def __repr__(self):
        return f'Network({",".join((str(len(l)) for l in self.layers))}, name={self.name})'

__all__ = ['Network']
