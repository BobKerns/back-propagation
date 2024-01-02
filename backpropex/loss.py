
from typing import Protocol
import numpy as np

class LossFunction(Protocol):
    """
    A cost function for a neural network.
    """
    name: str

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        """
        Evaluate the cost function for a given set of actual and expected values.
        """
        ...

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        """
        Evaluate the derivative of the cost function for a given set of actual and expected values.
        """
        ...

class MSE(LossFunction):
    """
    The mean squared error cost function.
    """
    name: str = 'MSE'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.square(actual - expected)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return 2 * (actual - expected) / len(actual)
