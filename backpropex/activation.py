
import math
from typing import Protocol


class ActivationFunction(Protocol):
    """
    An activation function and its derivative.
    """
    name: str

    @classmethod
    def __call__(cls, x: float) -> float:
        """Nonlinear activation response of a node."""
        ...

    @classmethod
    def derivative(cls, x: float) -> float:
        """Derivative of the activation function."""
        ...

class ACT_ReLU(ActivationFunction):
    """The standard ReLU activation function."""
    name = 'ReLU'

    @classmethod
    def __call__(cls, x: float) -> float:
        return max(0.0, x)

    @classmethod
    def derivative(cls, x: float) -> float:
        return 1.0 if x > 0.0 else 0.0

class ACT_Softmax(ActivationFunction):
    """The standard softmax activation function."""
    name = 'Softmax'

    @classmethod
    def __call__(cls, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @classmethod
    def derivative(cls, x: float) -> float:
        return cls(x) * (1.0 - cls(x))

class ACT_Sigmoid(ActivationFunction):
    """The standard sigmoid activation function."""
    name = 'Sigmoid'

    @classmethod
    def __call__(cls, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @classmethod
    def derivative(cls, x: float) -> float:
        return cls(x) * (1.0 - cls(x))
