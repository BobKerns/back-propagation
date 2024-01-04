"""
Loss functions and their derivatives.

Most of these functions are taken from the Keras documentation:
keras.io/api/losses/

Or at least, that is what Copilot says.  I have not verified this.
Copilot also says that the functions are taken from the TensorFlow documentation:
https://www.tensorflow.org/api_docs/python/tf/keras/losses

I have not verified this either.

The functions are also taken from the PyTorch documentation:
https://pytorch.org/docs/stable/nn.html#loss-functions

I have not verified this either.

The functions are also taken from the SciPy documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html

I have not verified this either.

The functions are also taken from the NumPy documentation:
https://numpy.org/doc/stable/reference/generated/numpy.square.html

I have not verified this either. But now I am starting to wonder if Copilot is just listing
every documentation source it can find.

The functions are also taken from the Wikipedia page on loss functions:
https://en.wikipedia.org/wiki/Loss_function

I have not verified this either.

The functions are also taken from the Wikipedia page on cross entropy:
https://en.wikipedia.org/wiki/Cross_entropy

I have not verified this either.

The functions are also taken from the Wikipedia page on Kullback-Leibler divergence:

https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

I have not verified this either.

The functions are also taken from the Wikipedia page on mean squared error:
https://en.wikipedia.org/wiki/Mean_squared_error

I have not verified this either.

The functions are also taken from the Wikipedia page on mean absolute error:

https://en.wikipedia.org/wiki/Mean_absolute_error

I have not verified this either.

The functions are also taken from the Wikipedia page on mean absolute percentage error:

https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

I have not verified this either.

The functions are also taken from the Wikipedia page on mean absolute logarithmic error:

https://en.wikipedia.org/wiki/Mean_absolute_logarithmic_error

I have not verified this either.

The functions are also taken from the Wikipedia page on mean squared logarithmic error:

https://en.wikipedia.org/wiki/Mean_squared_logarithmic_error

I have not verified this either.

Etc.

"""
from typing import Any, Protocol, TYPE_CHECKING, Self
import numpy as np
from typing import cast
type NPArray[T: Any] = np.ndarray[int, T]

class LossFunction(Protocol):
    """
    A cost function for a neural network.
    """
    name: str

    _registry: dict[str, 'LossFunction'] = {}
    @staticmethod

    def register(lossFn: 'LossFunction') -> 'LossFunction':
        """
        Register a new loss function.
        """
        LossFunction._registry[lossFn.name] = lossFn
        return lossFn

    @staticmethod

    def names() -> list[str]:
        """
        Return a list of the names of all registered loss functions.
        """
        return list(LossFunction._registry.keys())

    @classmethod
    def __call__(cls, actual: NPArray[float], expected: NPArray) -> float:
        """
        Evaluate the cost function for a given set of actual and expected values.
        """
        ...

    @classmethod
    def derivative(cls, actual: NPArray, expected: NPArray) -> NPArray:
        """
        Evaluate the derivative of the cost function for a given set of actual and expected values.
        """
        ...

    @classmethod

    def __getattr__(cls, name: str) -> Any:
        """
        Allow accessing loss functions via attribute access.
        """
        return LossFunction._registry[name]

    def __init_subclass__(cls) -> None:
        if hasattr(cls, 'name'):
            LossFunction.register(cls)
        return super().__init_subclass__()

class LossBase(LossFunction):
    """Base class to allow defining LossFunction objects via class syntax."""
    name: str
    def derivative(self, actual: NPArray, expected: NPArray) -> float:
        """Evaluate the derivative of the cost function for a given set of actual and expected values."""
        ...

    def __call__(self, actual: NPArray[float], expected: NPArray) -> float:
        """
        Evaluate the cost function for a given set of actual and expected values.
        """
        ...

    def __new__(cls, actual: NPArray, expected: NPArray) -> float:
        """Instead of instantiating, evaluate the cost function for a given set of actual and expected values."""
        return cls.__call__(cast(Self, cls), actual, expected)

    def __init_subclass__(cls) -> None:
        """Wrap the derivative method to allow defining LossFunction objects via class syntax."""
        sub = super().__init_subclass__()
        derivative = cls.derivative
        (sub or cls).derivative = lambda self, actual, expected: derivative(cast(Self, self), actual, expected)
        return sub

class MSE(LossBase):
    """
    The mean squared error cost function.
    """
    name: str = 'MSE'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.square(actual - expected)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return 2 * (actual - expected) / len(actual)

class CrossEntropy(LossBase):
    """
    The cross entropy cost function.
    """
    name: str = 'CrossEntropy'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, -np.sum(expected * np.log(actual) + (1.0 - expected) * np.log(1.0 - actual))) / len(actual)


    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return (actual - expected) / (actual * (1 - actual)) / len(actual)


class BinaryCrossEntropy(LossBase):
    """
    The binary cross entropy cost function.
    """
    name: str = 'BinaryCrossEntropy'


    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, -np.sum(expected * np.log(actual) + (1 - expected) * np.log(1 - actual))) / len(actual)


    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return (actual - expected) / (actual * (1 - actual)) / len(actual)

class CategoricalCrossEntropy(LossBase):
    """
    The categorical cross entropy cost function.
    """
    name: str = 'CategoricalCrossEntropy'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return -cast(float, np.sum(expected * np.log(actual))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual / len(actual)

class SparseCategoricalCrossEntropy(LossBase):
    """
    The sparse categorical cross entropy cost function.
    """
    name: str = 'SparseCategoricalCrossEntropy'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return -cast(float, np.sum(expected * np.log(actual))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual / len(actual)

class KLDivergence(LossBase):
    """
    The Kullback-Leibler divergence cost function.
    """
    name: str = 'KLDivergence'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual / len(actual)
class KLDivergenceCategorical(LossBase):
    """
    The Kullback-Leibler divergence cost function for categorical distributions.
    """
    name: str = 'KLDivergenceCategorical'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual / len(actual)

class KLDivergenceMultinomial(LossBase):
    """
    The Kullback-Leibler divergence cost function for multinomial distributions.
    """
    name: str = 'KLDivergenceMultinomial'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual / len(actual)

class KLDivergencePoisson(LossBase):
    """
    The Kullback-Leibler divergence cost function for Poisson distributions.
    """
    name: str = 'KLDivergencePoisson'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(expected * np.log(expected / actual) - expected + actual) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual + 1 / len(actual)

class KLDivergenceUniform(LossBase):
    """
    The Kullback-Leibler divergence cost function for uniform distributions.
    """
    name: str = 'KLDivergenceUniform'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.log(expected / actual)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual / len(actual)

class KLDivergenceWeighted(LossBase):
    """
    The weighted Kullback-Leibler divergence cost function.
    """
    name: str = 'KLDivergenceWeighted'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual / len(actual)

class KLDivergenceWeightedBernoulli(LossBase):
    """
    The weighted Kullback-Leibler divergence cost function for Bernoulli distributions.
    """
    name: str = 'KLDivergenceWeightedBernoulli'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual) + (1 - expected) * np.log((1 - expected)) / (1 - actual))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual + (1 - expected) / (1 - actual) / len(actual)

class KLDivergenceWeightedCategorical(LossBase):
    """
    The weighted Kullback-Leibler divergence cost function for categorical distributions.
    """
    name: str = 'KLDivergenceWeightedCategorical'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual / len(actual)

class KLDivergenceWeightedGaussian(LossBase):
    """
    The weighted Kullback-Leibler divergence cost function for Gaussian distributions.
    """
    name: str = 'KLDivergenceWeightedGaussian'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, np.sum(np.log(expected / actual) + (np.square(actual) + np.square(expected)) / 2)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual / len(actual)

class KLDivergenceWeightedMultinomial(LossBase):
    """
    The weighted Kullback-Leibler divergence cost function for multinomial distributions.
    """
    name: str = 'KLDivergenceWeightedMultinomial'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual / len(actual)

class KLDivergenceWeightedPoisson(LossBase):
    """
    The weighted Kullback-Leibler divergence cost function for Poisson distributions.
    """
    name: str = 'KLDivergenceWeightedPoisson'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(expected * np.log(expected / actual) - expected + actual) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual + 1 / len(actual)

class KLDivergenceWeightedUniform(LossBase):
    """
    The weighted Kullback-Leibler divergence cost function for uniform distributions.
    """
    name: str = 'KLDivergenceWeightedUniform'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.log(expected / actual)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual / len(actual)

class MeanAbsoluteError(LossBase):
    """
    The mean absolute error cost function.
    """
    name: str = 'MeanAbsoluteError'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.abs(actual - expected)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return np.sign(actual - expected) / len(actual)

class MeanAbsolutePercentageError(LossBase):
    """
    The mean absolute percentage error cost function.
    """
    name: str = 'MeanAbsolutePercentageError'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, np.sum(np.abs(actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return np.sign(actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps) / len(actual)

class MeanAbsoluteLogarithmicError(LossBase):
    """
    The mean absolute logarithmic error cost function.
    """
    name: str = 'MeanAbsoluteLogarithmicError'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.abs(np.log(actual + 1) - np.log(expected + 1))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return np.sign(np.log(actual + 1) - np.log(expected + 1)) / (actual + 1) / len(actual)

class MeanSquaredLogarithmicError(LossBase):
    """
    The mean squared logarithmic error cost function.
    """
    name: str = 'MeanSquaredLogarithmicError'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.square(np.log(actual + 1) - np.log(expected + 1))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return 2 * (np.log(actual + 1) - np.log(expected + 1)) / (actual + 1) / len(actual)

class MeanSquaredPercentageError(LossBase):
    """
    The mean squared percentage error cost function.
    """
    name: str = 'MeanSquaredPercentageError'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.square((actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return 2 * (actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps) / len(actual)

class MeanSquaredError(LossBase):
    """
    The mean squared error cost function.
    """
    name: str = 'MeanSquaredError'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.square(actual - expected)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return 2 * (actual - expected) / len(actual)

class Poisson(LossBase):
    """
    The Poisson cost function.
    """
    name: str = 'Poisson'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(actual - expected * np.log(actual)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return 1 - expected / actual / len(actual)

class CosineSimilarity(LossBase):
    """
    The cosine similarity cost function.
    """
    name: str = 'CosineSimilarity'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, np.sum(actual * expected)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return expected / len(actual)

class Hinge(LossBase):
    """
    The hinge cost function.
    """
    name: str = 'Hinge'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.maximum(0, 1 - actual * expected)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / len(actual)

class SquaredHinge(LossBase):
    """
    The squared hinge cost function.
    """
    name: str = 'SquaredHinge'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.square(np.maximum(0, 1 - actual * expected))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -2 * expected * np.maximum(0, 1 - actual * expected) / len(actual)

class LogCosh(LossBase):
    """
    The log cosh cost function.
    """
    name: str = 'LogCosh'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.log(np.cosh(actual - expected))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return np.tanh(actual - expected) / len(actual)

class Huber(LossBase):
    """
    The Huber cost function.
    """
    name: str = 'Huber'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.where(np.abs(actual - expected) < 1, 0.5 * np.square(actual - expected), np.abs(actual - expected) - 0.5)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return np.where(np.abs(actual - expected) < 1, actual - expected, np.sign(actual - expected)) / len(actual)

class Log(LossBase):
    """
    The log cost function.
    """
    name: str = 'Log'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.log(1 + np.exp(-actual * expected))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / (1 + np.exp(actual * expected)) / len(actual)

class Quantile(LossFunction):
    """
    The quantile cost function.  This is a continuous approximation of the pinball loss function.
    Unlike most classes in this module, this class must be instantiated with a quantile value.
    """
    name: str = 'Quantile'
    q: float

    def __init__(self, q: float = 0.5):
        self.q = q

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.where(actual >= expected, self.q * (actual - expected), (1 - self.q) * (expected - actual))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return np.where(actual >= expected, self.q, 1 - self.q) / len(actual)

class LogPoisson(LossBase):
    """
    The log Poisson cost function.
    """
    name: str = 'LogPoisson'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.exp(actual) - actual * expected) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return np.exp(actual) - expected / len(actual)

class KLDivergenceGaussian(LossBase):
    """
    The Kullback-Leibler divergence cost function for Gaussian distributions.
    """
    name: str = 'KLDivergenceGaussian'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, np.sum(np.log(expected / actual) + (np.square(actual) + np.square(expected)) / 2)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual / len(actual)

class PoissonNLL(LossBase):
    """
    The Poisson negative log likelihood cost function.
    """
    name: str = 'PoissonNLL'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(actual - expected * np.log(actual)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return 1 - expected / actual / len(actual)

class KLDivergenceBernoulli(LossBase):
    """
    The Kullback-Leibler divergence cost function for Bernoulli distributions.
    """
    name: str = 'KLDivergenceBernoulli'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual) + (1 - expected) * np.log((1 - expected) / (1 - actual)))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / actual + (1 - expected) / (1 - actual) / len(actual)

class CategoricalHinge(LossBase):
    """
    The categorical hinge cost function.
    """
    name: str = 'CategoricalHinge'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.maximum(0, 1 - actual * expected)) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return -expected / len(actual)

class SquaredLog(LossBase):
    """
    The squared log cost function.
    """
    name: str = 'SquaredLog'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.square(np.log(actual + 1) - np.log(expected + 1))) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return 2 * (np.log(actual + 1) - np.log(expected + 1)) / (actual + 1) / len(actual)

class PoissonNLLLog(LossBase):
    """
    The Poisson negative log likelihood cost function.
    """
    name: str = 'PoissonNLLLog'

    def __call__(self, actual: NPArray, expected: NPArray) -> float:
        return np.sum(np.exp(actual) - actual * expected) / len(actual)

    def derivative(self, actual: NPArray, expected: NPArray) -> NPArray:
        return np.exp(actual) - expected / len(actual)
