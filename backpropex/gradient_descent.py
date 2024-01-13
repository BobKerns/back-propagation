"""
A module to perform gradient descent on a neural network.
"""

from typing import Any, Generator, Optional
from backpropex.steps import StepType, TrainOptimizeStepResult
from backpropex.types import TrainingProgress
from backpropex.protocols import NetProtocol, OptimizerProtocol

class GradientDescent(OptimizerProtocol):
    """
    A gradient descent optimizer.
    """
    def __init__(self, learning_rate: float = 0.1) -> None:
        """
        Initialize the gradient descent optimizer.

        :param learning_rate: The learning rate.
        """
        self.learning_rate = learning_rate

    def __call__(self, net: NetProtocol, loss: float, /, *,
                 training_info: TrainingProgress,
                 learning_rate: Optional[float] = 0.1,
                 **kwargs: Any) -> Generator[TrainOptimizeStepResult, Any, None]:
        """
        Perform gradient descent on the network.

        :param net: The network to train.
        :param info: Information about the training step.
        """
        if learning_rate is not None:
            ...
        output = net.output_layer
        output.loss.fill(loss/len(output.loss))
        for from_layer, to_layer in zip(reversed(net.layers[0:-1]),
                                        reversed(net.layers[1:])):
            to_loss = to_layer.loss
            from_loss = from_layer.loss
            from_loss.fill(0.0)
            delta = to_layer.loss_delta
            # Update the weights, and track how much of the loss we've accounted for
            for to_node in to_layer.real_nodes:
                for edge in to_node.edges_to:
                    to_idx = to_node.idx
                    d = to_loss[to_idx] * to_layer.gradient[to_idx] * self.learning_rate
                    edge.weight -= d
                    delta[to_idx] += d
            # Propagate the loss back to the previous layer
            for to_node in to_layer.real_nodes:
                for edge in to_node.edges_to:
                    from_idx = edge.from_.idx
                    from_loss[from_idx] += delta[edge.to_.idx] * edge.weight
            with net.step_active(to_layer):
                yield TrainOptimizeStepResult(StepType.TrainOptimize,
                                         layer=to_layer,
                                            loss = loss * to_layer.gradient * self.learning_rate,
                                            weight_delta = to_layer.gradient * self.learning_rate,
                                            **training_info
                                            )
