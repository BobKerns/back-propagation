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
import re

import numpy as np

from backpropex.types import (
    EvalOutputStepResult,
    EvalStepResultAny,
    StepType,
    EvalForwardStepResult, EvalInputStepResult, LayerType, NPFloats, FloatSeq, NetTuple,
    TrainForwardStepResult,
    TrainInputStepResult,
    TrainLossStepResult,
    TrainOutputStepResult,
    TrainStepResultAny,
    TrainingData,
    TrainingInfo
)
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
class Network:
    """
    A neural network.
    """
    graph: DiGraph = DiGraph()
    layers: Sequence[Layer]
    loss_function: LossFunction
    max_layer_size: int
    name: str
    input_type: NetTuple
    output_type: NetTuple

    # Progress information for drawing the network
    # The layer that is currently being evaluated
    active_layer: Optional[Layer] = None
    active_message: Optional[str] = None
    # The item within the training set currently being trained
    datum_number: Optional[int] = None
    datum_max: int = 0
    datum_value: Optional[NPFloats] = None
    datum_expected: Optional[NPFloats] = None
    # The epoch currently being trained (pass through the training set))
    epoch_number: Optional[int] = None
    # The number of epochs to train
    epoch_max: int = 0
    # The loss for the current training item.
    loss: Optional[float] = None

    def __init__(self, *layers: int,
                 name: Optional[str] = None,
                 loss_function: LossFunction=MeanSquaredError,
                 activations: Optional[Sequence[ActivationFunction]]=None,
                 input_names: Optional[Sequence[str]]=None,
                 output_names: Optional[Sequence[str]]=None,
                 ):
        """
        A neural network.

        :param layers: The number of nodes in each layer.
        :param loss_function: The loss function for this network.
        :param activation_functions: The activation function for each layer.
        """
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

    @property
    def weights(self) -> Generator[float, None, None]:
        return (edge.weight for edge in self.edges)

    @property
    def values(self) -> Generator[float, None, None]:
        return (node.value for node in self.nodes)

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
    def training_epoch(self, epoch: int, epoch_max: int):
        """
        Set the active layer for the network during a training pass.
        """
        self.epoch_number = epoch
        self.epoch_max = epoch_max
        yield epoch
        self.epoch_number = None
        self.epoch_max = 0

    @contextmanager
    def training_datum(self, datum_number: int, datum_max: int,
             datum_value: NPFloats, datum_expected: NPFloats):
        """
        Set the active layer for the network during a training pass.
        """
        self.datum_number = datum_number
        self.datum_max = datum_max
        self.datum_value = datum_value
        self.datum_expected = datum_expected
        yield datum_number, datum_max, datum_value, datum_expected
        self.datum_number = None
        self.datum_max = 0
        self.datum_value = None
        self.datum_expected = None

    @contextmanager
    def training_loss(self,  output: NPFloats, expected: NPFloats, /):
        """
        Set the loss for the network during a training pass.
        """
        loss = self.loss_function(output, expected)
        self.loss = loss
        yield loss
        self.loss = None

    def __call__(self, input: FloatSeq, /, *,
                 label: Optional[str] = None
                 ) -> Generator[EvalStepResultAny, Any, None]:
        """
        Evaluate the network for a given input. Returns a generator that produces
        diagrams of the network as it is evaluated. The final value is the output
        from the network as a named tuple.
        """
        layer = self.layers[0]
        layer.values = input
        with self.step_active(layer):
            in_tuple = self.input_type(*input)
            yield EvalInputStepResult(StepType.Input, layer=layer, input=in_tuple)
        for layer in self.layers[1:]:
            with self.step_active(layer):
                for node in layer.real_nodes:
                    value = sum(edge.weight * edge.previous.value for edge in self.edges)
                    node.value = node.activation(value)
                yield EvalForwardStepResult(StepType.Forward, layer=layer)
        # Yeld the result back to the caller.
        # We need a better protocol for this.
        yield EvalOutputStepResult(StepType.Output,
                                   layer=self.output_layer,
                                   output=self.output_type(*self.output))

    def train_one(self, input: FloatSeq, expected: FloatSeq, /,
                  datum_number: int = 0, datum_max: int = 1,
                  ) -> Generator[TrainStepResultAny, Any, None]:
        """
        Train the network for a given input and expected output.
        """
        input = np.array(input)
        expected = np.array(expected)
        with self.training_datum(datum_number, datum_max, input, expected):
            training_info: TrainingInfo = TrainingInfo(epoch=self.epoch_number or 0,
                                         epoch_max=self.epoch_max,
                                         datum_no=datum_number,
                                         datum_max=datum_max,
                                         datum=(input, expected))
            # Forward pass
            def map_step(step: EvalStepResultAny):
                """Extemd the eval step with training info."""
                match(step):
                    case EvalInputStepResult():
                        return TrainInputStepResult(StepType.TrainInput,
                                                    layer=step.layer,
                                                    input=step.input,
                                                    **training_info)
                    case EvalForwardStepResult():
                        return TrainForwardStepResult(StepType.TrainForward,
                                                    layer=step.layer,
                                                    **training_info)
                    case EvalOutputStepResult():
                        return TrainOutputStepResult(StepType.TrainOutput,
                                                    layer=step.layer,
                                                    output=step.output,
                                                    **training_info)
            yield from (map_step(r) for r in self(input))
            # Backward pass
            layer = self.layers[-1]
            output = self.output_array
            loss = self.loss_function(output, expected)
            with self.training_loss(output, expected) as loss:
                yield TrainLossStepResult(StepType.TrainLoss, layer=layer, loss=loss, **training_info)
                for layer in reversed(self.layers[0:-1]):
                    for node in layer.real_nodes:
                        value = sum(edge.weight * edge.next.value for edge in self.edges)
                        print(f'TODO: Node {node.idx} value={value:.2f}')

    def backpropagate(self, output: NPFloats, expected: NPFloats, /):
        """
        Backpropagate the gradient through the network.
        """
        if self.datum_expected is None:
            raise ValueError('No expected output set')
        with self.training_loss(output, expected) as loss:
            grad = self.loss_function.derivative(output, expected)
            print(f'Loss={loss:.2f}, grad={grad:.2f}')
            # Backward pass
            layer = self.layers[-1]
            for node in layer.real_nodes:
                node.gradient = node.value - self.datum_expected[node.idx]
            for layer in reversed(self.layers[0:-1]):
                for node in layer.real_nodes:
                    total_weights = sum([
                        edge.weight * (edge.next.gradient or 0.0)
                        for edge
                        in self.in_edges(node)
                    ], 0.0)
                    d = node.activation.derivative(node.value)
                    gradient = (
                        cast(float, d * edge.weight / total_weights)
                        for edge
                        in self.in_edges(node)
                    )
                    node.gradient = np.array(gradient)

    def train(self, data: TrainingData, /, *,
              epochs: int=1000,
              learning_rate: float=0.1
              ) -> Generator[TrainStepResultAny, Any, None]:
        """
        Train the network for a given set of data.
        """
        datum_max = len(data)
        for epoch in range(epochs):
            with self.training_epoch(epoch, epochs):
                for idx, (input, expected) in enumerate(data):
                    input = np.array(input)
                    expected = np.array(expected)
                    with self.training_datum(idx, datum_max, input, expected):
                        yield from self.train_one(input, expected)

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
        """The output values of this network as a namedtuple."""
        return self.output_type(*(n.value for n in self.output_layer.real_nodes))

    @property
    def output_array(self):
        """The output values of this network as an array."""
        return np.array([n.value for n in self.output_layer.real_nodes])
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
