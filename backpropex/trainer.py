"""
This module contains the Trainer class, which is used to train a neural network.

The Trainer class is a wrapper. It wraps a neural network and is called on a training
data set to train the network.
"""

from contextlib import contextmanager
from typing import Any, Generator, Optional, cast

import numpy as np

from backpropex.loss import LossFunction, MeanSquaredError
from backpropex.types import (
    EvalForwardStepResult, EvalInputStepResult, EvalOutputStepResult,
    EvalStepResultAny, FloatSeq, NPFloats,
    NetProtocol, TrainProtocol,
    StepType,
    InitStepResult, TrainForwardStepResult, TrainInputStepResult,
    TrainLossStepResult, TrainOutputStepResult, TrainStepResultAny,
    TrainingData, TrainingInfo,
)


class Trainer(TrainProtocol):
    """
    The trainer for a neural network.
    """
    learning_rate: float
    loss_function: LossFunction

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

    def __init__(self, network: NetProtocol, /, *, loss_function: LossFunction = MeanSquaredError):
        self.net = network
        self.loss_function = loss_function

    def __call__(self, data: TrainingData, /, *,
                    epochs: int=1, batch_size: int=1,
                    learning_rate: float=0.1) -> Generator[TrainStepResultAny|InitStepResult, Any, None]:
        """
        Train the network on the given training data.

        The training data is a sequence of tuples, where each tuple contains
        the input and the expected output of the network.

        The training is done in epochs. Each epoch is a complete pass through
        the training data. The training data is shuffled before each epoch.

        The training data is processed in batches. The batch size is the number
        of training data tuples that are processed before the weights are updated.
        """
        datum_max = len(data)
        for epoch in range(epochs):
            with self.training_epoch(epoch, epochs):
                for idx, (input, expected) in enumerate(data):
                    input = np.array(input)
                    expected = np.array(expected)
                    with self.training_datum(idx, datum_max, input, expected):
                        yield from self.train_one(input, expected)


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
                    case InitStepResult():
                        return step
            yield from (map_step(r) for r in self.net(input))
            # Backward pass
            layer = self.net.layers[-1]
            output = self.net.output_array
            loss = self.loss_function(output, expected)
            with self.training_loss(output, expected) as loss:
                yield TrainLossStepResult(StepType.TrainLoss, layer=layer, loss=loss, **training_info)
                for layer in reversed(self.net.layers[0:-1]):
                    for node in layer.real_nodes:
                        value = sum(edge.weight * edge.next.value for edge in self.net.edges)
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
            layer = self.net.layers[-1]
            for node in layer.real_nodes:
                node.gradient = node.value - self.datum_expected[node.idx]
            for layer in reversed(self.net.layers[0:-1]):
                for node in layer.real_nodes:
                    total_weights = sum([
                        edge.weight * (edge.next.gradient or 0.0)
                        for edge
                        in self.net.in_edges(node)
                    ], 0.0)
                    d = node.activation.derivative(node.value)
                    gradient = (
                        cast(float, d * edge.weight / total_weights)
                        for edge
                        in self.net.in_edges(node)
                    )
                    node.gradient = np.array(gradient)
                    print(f'TODO: Node {node.idx} gradient={node.gradient}')
