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

from networkx import DiGraph
from typing import Any, Generator, NamedTuple, Optional, Sequence, cast

import numpy as np
from backpropex.builder import DefaultBuilder, sanitize

from backpropex.steps import (
    EvalOutputStepResult,
    EvalStepResultAny,
    InitStepResult,
    StepType,
    EvalForwardStepResult, EvalInputStepResult,
)
from backpropex.types import (
    FloatSeq, NetTuple,
)
from backpropex.protocols import Builder, NetProtocol
from backpropex.edge import Edge
from backpropex.layer import Layer
from backpropex.node import Node

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
    graph: DiGraph = DiGraph()
    input_type: NetTuple
    output_type: NetTuple
    layers: Sequence[Layer]
    max_layer_size: int
    name: str

    def __init__(self, *layers: int,
                 name: Optional[str] = None,
                 builder: Optional[Builder|type[Builder]] = DefaultBuilder,
                 **kwargs: Any
                 ):
        """
        A neural network.

        :param layers: The number of nodes in each layer.
        :param loss_function: The loss function for this network.
        :param activation_functions: The activation function for each layer.
        """
        self.net = self

        self.name = name if name is not None else f'Network_{next(ids)}'

        if isinstance(builder, type):
            builder()(self, *layers, **kwargs)
        elif isinstance(builder, Builder):
            builder(self, *layers, **kwargs)
        else:
            raise TypeError(f'Invalid builder type: {builder}')

        self.max_layer_size = max(len(layer) for layer in self.layers)
        self.input_type = self.mk_namedtuple('input', self.input_layer)
        self.output_type = self.mk_namedtuple('output', self.output_layer)


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

    def __call__(self, input: FloatSeq, /, *,
                 label: Optional[str] = None
                 ) -> Generator[EvalStepResultAny, Any, None]:
        """
        Evaluate the network for a given input. Returns a generator that produces
        diagrams of the network as it is evaluated. The final value is the output
        from the network as a named tuple.
        """
        yield InitStepResult(StepType.Initialized)
        layer = self.layers[0]
        layer.values = input
        with self.step_active(layer):
            in_tuple = self.input_type(*input)
            yield EvalInputStepResult(StepType.Input, layer=layer, input=in_tuple)
        for layer in self.layers[1:]:
            with self.step_active(layer):
                for node in layer.real_nodes:
                    value = sum(edge.weight * edge.previous.value for edge in self.in_edges(node))
                    node.value = node.activation(value)
                yield EvalForwardStepResult(StepType.Forward,
                                            layer=layer,
                                            values=layer.values,
                                            )
        # Yeld the result back to the caller.
        # We need a better protocol for this.
        yield EvalOutputStepResult(StepType.Output,
                                   layer=self.output_layer,
                                   output=self.output_type(*self.output))
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
    def labels(self) -> dict[Node, str]:
        return {n: n.label for n in self.nodes}

    @property
    def weights(self) -> Generator[float, None, None]:
        return (edge.weight for edge in self.edges)

    @property
    def values(self) -> Generator[float, None, None]:
        return (node.value for node in self.nodes)
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
