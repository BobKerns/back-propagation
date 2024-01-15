"""
This module contains the Trainer class, which is used to train a neural network.

The Trainer class is a wrapper. It wraps a neural network and is called on a training
data set to train the network.
"""

from contextlib import contextmanager
from collections.abc import Generator
from typing import Any, Optional
from random import shuffle

import numpy as np
from backpropex.backpropagate import Backpropagate
from backpropex.gradient_descent import GradientDescent

from backpropex.loss import LossFunction, MeanSquaredError
from backpropex.steps import (
    StepType, InitStepResult,
    EvalForwardStepResult, EvalInputStepResult, EvalOutputStepResult, StepTypeAny,
    TrainForwardStepResult, TrainInputStepResult,
    TrainLossStepResult,TrainOutputStepResult,
    EvalStepResultAny, TrainStepResultAny, TrainStepType,
)
from backpropex.types import (
    NPFloat1D,
    TrainingData, TrainingProgress, TrainingItem,
    null_training_item,
)
from backpropex.protocols import (
    BackpropagateProtocol, Filter, NetProtocol, OptimizerProtocol, TrainProtocol, Trace,
)
from backpropex.utils import make

class Trainer(TrainProtocol):
    """
    The trainer for a neural network.
    """
    learning_rate: float
    loss_function: LossFunction

    training_progress: TrainingProgress

    _filter: Optional[Filter|type[Filter]] = None
    _trace: Optional[Trace|type[Trace]] = None
    _backpropagate: Optional[BackpropagateProtocol] = None

    @property
    def backpropagate(self) -> BackpropagateProtocol:
        if self._backpropagate is None:
            self._backpropagate = Backpropagate()
        return self._backpropagate

    _optimize: Optional[OptimizerProtocol] = None
    @property
    def optimize(self) -> OptimizerProtocol:
        return self._optimize or GradientDescent()

    def __init__(self, network: NetProtocol, /, *,
                 loss_function: LossFunction = MeanSquaredError,
                 learning_rate: float = 0.1,
                 filter: Optional[Filter|type[Filter]] = None,
                 trace: Optional[Trace|type[Trace]] = None,
                 optimizer: Optional[OptimizerProtocol|type[OptimizerProtocol]] = None,
                 ):
        self.net = network
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        if filter is not None:
            self._filter = make(filter, Filter)
        if trace is not None:
            self._trace = make(trace, Trace)
        self.training_progress = TrainingProgress(
            epoch=0, epoch_max=0, datum_no=0, datum_max=0,
            output_loss=0.0,
            datum=null_training_item
        )

    def __call__(self, data: TrainingData, /, *,
                    epochs: int=1, batch_size: int=1,
                    learning_rate: Optional[float] = None,
                    filter: Optional[Filter|type[Filter]] = None,
                    trace: Optional[Trace|type[Trace]] = None,
                    **kwargs: Any
                    ) -> Generator[TrainStepResultAny, Any, None]:
        """
        Train the network on the given training data.

        The training data is a sequence of tuples, where each tuple contains
        the input and the expected output of the network.

        The training is done in epochs. Each epoch is a complete pass through
        the training data. The training data is shuffled before each epoch.

        The training data is processed in batches. The batch size is the number
        of training data tuples that are processed before the weights are updated.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        # Copy the data, since we'll be shuffling it
        tdata = [TrainingItem(np.array(input, np.float_),
                              np.array(expected, np.float_), id)
                for id, (input, expected) in enumerate(data)]
        with self.net.trace(trace):
            with self.net.filter(filter):
                with self.net.filter(self._filter):
                    datum_max = len(tdata)
                    for epoch in range(epochs):
                        with self.training_epoch(epoch, epochs):
                            shuffle(tdata)
                            for idx, datum in enumerate(tdata):
                                with self.training_datum(idx, datum_max, datum):
                                    # Filter out the initialized steps after the first epoch/datum
                                     yield from (
                                        step
                                        for step in self.train_one(datum,
                                                                    datum_number=idx,
                                                                    datum_max=datum_max,
                                                                    **kwargs
                                                                    )
                                        if (step.type != StepType.Initialized
                                            or (epoch == 0 and idx == 0))
                                    )

    @contextmanager
    def training_epoch(self, epoch: int, epoch_max: int):
        """
        Set the active layer for the network during a training pass.
        """
        tp = self.training_progress
        tp['epoch'] = epoch
        tp['epoch_max'] = epoch_max
        yield tp

    @contextmanager
    def training_datum(self, datum_number: int, datum_max: int,
             datum: TrainingItem):
        """
        Set the active layer for the network during a training pass.
        """
        tp = self.training_progress
        tp['datum_no'] = datum_number
        tp['datum_max'] = datum_max
        tp['datum'] = datum
        yield self.training_progress

    @contextmanager
    def training_loss(self,  output: NPFloat1D, expected: NPFloat1D, /):
        """
        Set the loss for the network during a training pass.
        """
        loss = self.loss_function(output, expected)
        self.training_progress['output_loss'] = loss
        yield loss


    def train_one(self, datum: TrainingItem, /, *,
                  datum_number: int = 0,
                  datum_max: int = 1,
                  **kwargs: Any
                  ) -> Generator[TrainStepResultAny, Any, None]:
        """
        Train the network for a given input and expected output.
        """
        input = np.array(datum.input)
        expected = np.array(datum.expected)
        with self.training_datum(datum_number, datum_max, datum) as training_progress:
            # Forward pass
            def map_step(step: EvalStepResultAny):
                """Extemd the eval step with training info."""
                match(step):
                    case EvalInputStepResult():
                        return TrainInputStepResult(StepType.TrainInput,
                                                    layer=step.layer,
                                                    input=step.input,
                                                    **training_progress)
                    case EvalForwardStepResult():
                        return TrainForwardStepResult(StepType.TrainForward,
                                                    layer=step.layer,
                                                    **training_progress)
                    case EvalOutputStepResult():
                        return TrainOutputStepResult(StepType.TrainOutput,
                                                    layer=step.layer,
                                                    output=step.output,
                                                    **training_progress)
                    case InitStepResult():
                        return step
            def map_step_type(type: StepTypeAny) -> TrainStepType:
                """Map the step type."""
                match(type):
                    case StepType.Input:
                        return StepType.TrainInput
                    case StepType.Forward:
                        return StepType.TrainForward
                    case StepType.Output:
                        return StepType.TrainOutput
                    case StepType.Initialized:
                        return StepType.Initialized
                    case _:
                        raise ValueError(f'Unexpected step type {type}')
            for step in self.net(input):
                yield from self.net.filterCheck(map_step_type(step.type), lambda: map_step(step))
            # Backward s
            layer = self.net.layers[-1]
            output = self.net.output_array
            loss = self.loss_function(output, expected)
            with self.training_loss(output, expected) as loss:
                yield TrainLossStepResult(StepType.TrainLoss, layer=layer, loss=loss, **training_progress)
                yield from self.backpropagate(self, datum,
                                              training_progress=training_progress,
                                              **kwargs)
                yield from self.optimize(self.net, loss,
                                         training_progress=training_progress,
                                         **kwargs)
