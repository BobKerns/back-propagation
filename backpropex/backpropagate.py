"""
Backpropagate the gradient through the network.
"""

from typing import Any, Generator
from backpropex.protocols import BackpropagateProtocol, TrainProtocol
from backpropex.steps import StepType, TrainBackwardStepResult
from backpropex.types import TrainingProgress, TrainingItem

class Backpropagate(BackpropagateProtocol):
    """Backpropagate the gradient through the network."""
    def __call__(self, trainer: TrainProtocol, training_item: TrainingItem, /, *,
                 training_info: TrainingProgress,
                 **kwargs: Any) -> Generator[TrainBackwardStepResult, Any, None]:
        """
        Backpropagate the gradient through the network.
        """
        net = trainer.net
        expected = training_item.expected
        grad = trainer.loss_function.derivative(net.output_array, expected, **kwargs)
        net.layers[-1].gradient = grad
        # Backward pass
        for from_layer, to_layer in zip(reversed(net.layers[0:-1]),
                                        reversed(net.layers[1:])):
            from_gradient = from_layer.gradient
            from_gradient.fill(0.0)
            to_gradient = to_layer.gradient
            for to_node in to_layer.real_nodes:
                d = to_node.activation.derivative(to_node.value)
                for edge in to_node.edges_to:
                    to_grad = to_gradient[to_node.idx]
                    from_gradient[edge.from_.idx] += d * to_grad * edge.weight
            with net.step_active(to_layer):
                yield TrainBackwardStepResult(StepType.TrainBackward,
                                            layer=to_layer,
                                            gradient=to_gradient,
                                            **training_info
                                            )
