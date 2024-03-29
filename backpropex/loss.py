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

import numpy as np
from typing import Any, cast

from backpropex.types import NPFloat1D
from backpropex.protocols import LossFunction

_registry: dict[str, LossFunction] = {}
def register(af: 'LossFunction'):
    """
    Register an Loss function.
    """
    _registry[af.name] = af
    return af

def get(name: str) -> 'LossFunction|None':
    """
    Get an Loss function by name.
    """
    return _registry[name]

def names() -> list[str]:
    """
    Get the names of all Loss functions.
    """
    return list(_registry.keys())

def loss(cls: type[LossFunction]) -> LossFunction:
    """
    Decorator to register an Loss function.
    """
    af = cls()
    register(af)
    return af

@loss
class MSE(LossFunction):
    """
    The mean squared error cost function.
    """
    name: str = 'MSE'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(np.square(actual - expected))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D,
                   **kwargs: Any) -> NPFloat1D:
        return 2 * (actual - expected) / len(actual)

from numpy import log as nplog
@loss
class CrossEntropy(LossFunction):
    """
    The cross entropy cost function.
    """
    name: str = 'CrossEntropy'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D,
                 **kwargs: Any) -> float:
        return cast(float, -np.sum(expected * nplog(actual) + (1.0 - expected) * nplog(1.0 - actual))) / len(actual)


    def derivative(self, actual: NPFloat1D, expected: NPFloat1D,
                   **kwargs: Any) -> NPFloat1D:
        return (actual - expected) / (actual * (1 - actual)) / len(actual)


@loss
class BinaryCrossEntropy(LossFunction):
    """
    The binary cross entropy cost function.
    """
    name: str = 'BinaryCrossEntropy'


    def __call__(self, actual: NPFloat1D, expected: NPFloat1D,
                 **kwargs: Any) -> float:
        return cast(float, -np.sum(expected * np.log(actual) + (1 - expected) * np.log(1 - actual))) / len(actual)


    def derivative(self, actual: NPFloat1D, expected: NPFloat1D,
                   **kwargs: Any) -> NPFloat1D:
        return (actual - expected) / (actual * (1 - actual)) / len(actual)

@loss
class CategoricalCrossEntropy(LossFunction):
    """
    The categorical cross entropy cost function.
    """
    name: str = 'CategoricalCrossEntropy'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D,
                 **kwargs: Any) -> float:
        return -cast(float, np.sum(expected * np.log(actual))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D,
                   **kwargs: Any) -> NPFloat1D:
        return -expected / actual / len(actual)

@loss
class SparseCategoricalCrossEntropy(LossFunction):
    """
    The sparse categorical cross entropy cost function.
    """
    name: str = 'SparseCategoricalCrossEntropy'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D,
                 **kwargs: Any) -> float:
        return -cast(float, np.sum(expected * np.log(actual))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D,
                   **kwargs: Any) -> NPFloat1D:
        return -expected / actual / len(actual)

@loss
class KLDivergence(LossFunction):
    """
    The Kullback-Leibler divergence cost function.
    """
    name: str = 'KLDivergence'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual / len(actual)
@loss
class KLDivergenceCategorical(LossFunction):
    """
    The Kullback-Leibler divergence cost function for categorical distributions.
    """
    name: str = 'KLDivergenceCategorical'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual / len(actual)

@loss
class KLDivergenceMultinomial(LossFunction):
    """
    The Kullback-Leibler divergence cost function for multinomial distributions.
    """
    name: str = 'KLDivergenceMultinomial'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual / len(actual)

@loss
class KLDivergencePoisson(LossFunction):
    """
    The Kullback-Leibler divergence cost function for Poisson distributions.
    """
    name: str = 'KLDivergencePoisson'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual) - expected + actual)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual + 1 / len(actual)

@loss
class KLDivergenceUniform(LossFunction):
    """
    The Kullback-Leibler divergence cost function for uniform distributions.
    """
    name: str = 'KLDivergenceUniform'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.log(expected / actual)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual / len(actual)

@loss
class KLDivergenceWeighted(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function.
    """
    name: str = 'KLDivergenceWeighted'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual / len(actual)

@loss
class KLDivergenceWeightedBernoulli(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function for Bernoulli distributions.
    """
    name: str = 'KLDivergenceWeightedBernoulli'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual) + (1 - expected) * np.log((1 - expected)) / (1 - actual))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual + (1 - expected) / (1 - actual) / len(actual)

@loss
class KLDivergenceWeightedCategorical(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function for categorical distributions.
    """
    name: str = 'KLDivergenceWeightedCategorical'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual / len(actual)

@loss
class KLDivergenceWeightedGaussian(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function for Gaussian distributions.
    """
    name: str = 'KLDivergenceWeightedGaussian'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(np.log(expected / actual) + (np.square(actual) + np.square(expected)) / 2)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual / len(actual)

@loss
class KLDivergenceWeightedMultinomial(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function for multinomial distributions.
    """
    name: str = 'KLDivergenceWeightedMultinomial'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual / len(actual)

@loss
class KLDivergenceWeightedPoisson(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function for Poisson distributions.
    """
    name: str = 'KLDivergenceWeightedPoisson'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual) - expected + actual)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual + 1 / len(actual)

@loss
class KLDivergenceWeightedUniform(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function for uniform distributions.
    """
    name: str = 'KLDivergenceWeightedUniform'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.log(expected / actual)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual / len(actual)

@loss
class MeanAbsoluteError(LossFunction):
    """
    The mean absolute error cost function.
    """
    name: str = 'MeanAbsoluteError'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.abs(actual - expected)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return np.sign(actual - expected) / len(actual)

@loss
class MeanAbsolutePercentageError(LossFunction):
    """
    The mean absolute percentage error cost function.
    """
    name: str = 'MeanAbsolutePercentageError'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(np.abs(actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return np.sign(actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps) / len(actual)

@loss
class MeanAbsoluteLogarithmicError(LossFunction):
    """
    The mean absolute logarithmic error cost function.
    """
    name: str = 'MeanAbsoluteLogarithmicError'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.abs(np.log(actual + 1) - np.log(expected + 1))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return np.sign(np.log(actual + 1) - np.log(expected + 1)) / (actual + 1) / len(actual)

@loss
class MeanSquaredLogarithmicError(LossFunction):
    """
    The mean squared logarithmic error cost function.
    """
    name: str = 'MeanSquaredLogarithmicError'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.square(np.log(actual + 1) - np.log(expected + 1))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return 2 * (np.log(actual + 1) - np.log(expected + 1)) / (actual + 1) / len(actual)

@loss
class MeanSquaredPercentageError(LossFunction):
    """
    The mean squared percentage error cost function.
    """
    name: str = 'MeanSquaredPercentageError'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.square((actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return 2 * (actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps) / len(actual)

@loss
class MeanSquaredError(LossFunction):
    """
    The mean squared error cost function.
    """
    name: str = 'MeanSquaredError'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.square(actual - expected)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return 2 * (actual - expected) / len(actual)

@loss
class Poisson(LossFunction):
    """
    The Poisson cost function.
    """
    name: str = 'Poisson'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(actual - expected * np.log(actual))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return 1 - expected / actual / len(actual)

@loss
class CosineSimilarity(LossFunction):
    """
    The cosine similarity cost function.
    """
    name: str = 'CosineSimilarity'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(actual * expected)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return expected / len(actual)

@loss
class Hinge(LossFunction):
    """
    The hinge cost function.
    """
    name: str = 'Hinge'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.maximum(0, 1 - actual * expected)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / len(actual)

@loss
class SquaredHinge(LossFunction):
    """
    The squared hinge cost function.
    """
    name: str = 'SquaredHinge'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.square(np.maximum(0, 1 - actual * expected))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -2 * expected * np.maximum(0, 1 - actual * expected) / len(actual)

@loss
class LogCosh(LossFunction):
    """
    The log cosh cost function.
    """
    name: str = 'LogCosh'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.log(np.cosh(actual - expected))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return np.tanh(actual - expected) / len(actual)

@loss
class Huber(LossFunction):
    """
    The Huber cost function.
    """
    name: str = 'Huber'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.where(np.abs(actual - expected) < 1, 0.5 * np.square(actual - expected), np.abs(actual - expected) - 0.5)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return np.where(np.abs(actual - expected) < 1, actual - expected, np.sign(actual - expected)) / len(actual)

@loss
class Log(LossFunction):
    """
    The log cost function.
    """
    name: str = 'Log'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.log(1 + np.exp(-actual * expected))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / (1 + np.exp(actual * expected)) / len(actual)

@loss
class Quantile(LossFunction):
    """
    The quantile cost function.  This is a continuous approximation of the pinball loss function.
    Unlike most classes in this module, this class must be instantiated with a quantile value.
    """
    name: str = 'Quantile'
    q: float

    def __init__(self, q: float = 0.5):
        self.q = q
        self.name = f'Quantile (q={q})'
        register(self)

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.where(actual >= expected, self.q * (actual - expected), (1 - self.q) * (expected - actual))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return np.where(actual >= expected, self.q, 1 - self.q) / len(actual)

@loss
class LogPoisson(LossFunction):
    """
    The log Poisson cost function.
    """
    name: str = 'LogPoisson'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(np.exp(actual) - actual * expected)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return np.exp(actual) - expected / len(actual)

@loss
class KLDivergenceGaussian(LossFunction):
    """
    The Kullback-Leibler divergence cost function for Gaussian distributions.
    """
    name: str = 'KLDivergenceGaussian'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(np.log(expected / actual) + (np.square(actual) + np.square(expected)) / 2)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual / len(actual)

@loss
class PoissonNLL(LossFunction):
    """
    The Poisson negative log likelihood cost function.
    """
    name: str = 'PoissonNLL'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(actual - expected * np.log(actual))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return 1 - expected / actual / len(actual)

@loss
class KLDivergenceBernoulli(LossFunction):
    """
    The Kullback-Leibler divergence cost function for Bernoulli distributions.
    """
    name: str = 'KLDivergenceBernoulli'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(expected * np.log(expected / actual) + (1 - expected) * np.log((1 - expected) / (1 - actual)))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / actual + (1 - expected) / (1 - actual) / len(actual)

@loss
class CategoricalHinge(LossFunction):
    """
    The categorical hinge cost function.
    """
    name: str = 'CategoricalHinge'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.maximum(0, 1 - actual * expected)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return -expected / len(actual)

@loss
class SquaredLog(LossFunction):
    """
    The squared log cost function.
    """
    name: str = 'SquaredLog'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return np.sum(np.square(np.log(actual + 1) - np.log(expected + 1))) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return 2 * (np.log(actual + 1) - np.log(expected + 1)) / (actual + 1) / len(actual)

@loss
class PoissonNLLLog(LossFunction):
    """
    The Poisson negative log likelihood cost function.
    """
    name: str = 'PoissonNLLLog'

    def __call__(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> float:
        return cast(float, np.sum(np.exp(actual) - actual * expected)) / len(actual)

    def derivative(self, actual: NPFloat1D, expected: NPFloat1D, /,
                 **kwargs: Any) -> NPFloat1D:
        return np.exp(actual) - expected / len(actual)

__all__ = [
    'MSE',
    'CrossEntropy',
    'BinaryCrossEntropy',
    'CategoricalCrossEntropy',
    'SparseCategoricalCrossEntropy',
    'KLDivergence',
    'KLDivergenceCategorical',
    'KLDivergenceMultinomial',
    'KLDivergencePoisson',
    'KLDivergenceUniform',
    'KLDivergenceWeighted',
    'KLDivergenceWeightedBernoulli',
    'KLDivergenceWeightedCategorical',
    'KLDivergenceWeightedGaussian',
    'KLDivergenceWeightedMultinomial',
    'KLDivergenceWeightedPoisson',
    'KLDivergenceWeightedUniform',
    'MeanAbsoluteError',
    'MeanAbsolutePercentageError',
    'MeanAbsoluteLogarithmicError',
    'MeanSquaredLogarithmicError',
    'MeanSquaredPercentageError',
    'MeanSquaredError',
    'Poisson',
    'CosineSimilarity',
    'Hinge',
    'SquaredHinge',
    'LogCosh',
    'Huber',
    'Log',
    'Quantile',
    'LogPoisson',
    'KLDivergenceGaussian',
    'PoissonNLL',
    'KLDivergenceBernoulli',
    'CategoricalHinge',
    'SquaredLog',
    'PoissonNLLLog',
    'register',
    'get',
    'names',
    'loss',
]
