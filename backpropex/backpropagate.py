"""
Backpropagate the gradient through the network.
"""

from collections.abc import Generator
from typing import Any
from backpropex.protocols import BackpropagateProtocol, TrainProtocol
from backpropex.steps import StepType, TrainBackwardStepResult
from backpropex.types import TrainingProgress, TrainingItem

class Backpropagate(BackpropagateProtocol):
    """Backpropagate the gradient through the network."""
    def __call__(self, trainer: TrainProtocol, training_item: TrainingItem, /, *,
                 training_progress: TrainingProgress,
                 **kwargs: Any) -> Generator[TrainBackwardStepResult, Any, None]:
        """
        Backpropagate the gradient through the network.
        """
        net = trainer.net
        expected = training_item.expected
        grad = trainer.loss_function.derivative(net.output_array, expected, **kwargs)
        net.layers[-1].gradient = grad
        # Backward pass
        for from_layer, to_layer in net.layer_pairs(reverse=True)
                yield TrainBackwardStepResult(StepType.TrainBackward,
                                            layer=to_layer,
                                            gradient=to_gradient,
                                            **training_progress
                                            )
