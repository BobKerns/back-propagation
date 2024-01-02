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

class CrossEntropy(LossFunction):
    """
    The cross entropy cost function.
    """
    name: str = 'CrossEntropy'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return -np.sum(expected * np.log(actual) + (1 - expected) * np.log(1 - actual)) / len(actual)
    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return (actual - expected) / (actual * (1 - actual)) / len(actual)

class BinaryCrossEntropy(LossFunction):
    """
    The binary cross entropy cost function.
    """
    name: str = 'BinaryCrossEntropy'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return -np.sum(expected * np.log(actual) + (1 - expected) * np.log(1 - actual)) / len(actual)

    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return (actual - expected) / (actual * (1 - actual)) / len(actual)

class CategoricalCrossEntropy(LossFunction):
    """
    The categorical cross entropy cost function.
    """
    name: str = 'CategoricalCrossEntropy'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return -np.sum(expected * np.log(actual)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class SparseCategoricalCrossEntropy(LossFunction):
    """
    The sparse categorical cross entropy cost function.
    """
    name: str = 'SparseCategoricalCrossEntropy'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return -np.sum(expected * np.log(actual)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class KLDivergence(LossFunction):
    """
    The Kullback-Leibler divergence cost function.
    """
    name: str = 'KLDivergence'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(expected * np.log(expected / actual)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class KLDivergenceBernoulli(LossFunction):
    """
    The Kullback-Leibler divergence cost function for Bernoulli distributions.
    """
    name: str = 'KLDivergenceBernoulli'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(expected * np.log(expected / actual) + (1 - expected) * np.log((1 - expected) / (1 - actual))) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual + (1 - expected) / (1 - actual) / len(actual)

class KLDivergenceCategorical(LossFunction):
    """
    The Kullback-Leibler divergence cost function for categorical distributions.
    """
    name: str = 'KLDivergenceCategorical'
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(expected * np.log(expected / actual)) / len(actual)

    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class KLDivergenceGaussian(LossFunction):
    """
    The Kullback-Leibler divergence cost function for Gaussian distributions.
    """
    name: str = 'KLDivergenceGaussian'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.log(expected / actual) + (np.square(actual) + np.square(expected)) / 2) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class KLDivergenceMultinomial(LossFunction):
    """
    The Kullback-Leibler divergence cost function for multinomial distributions.
    """
    name: str = 'KLDivergenceMultinomial'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(expected * np.log(expected / actual)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class KLDivergencePoisson(LossFunction):
    """
    The Kullback-Leibler divergence cost function for Poisson distributions.
    """
    name: str = 'KLDivergencePoisson'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(expected * np.log(expected / actual) - expected + actual) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual + 1 / len(actual)

class KLDivergenceUniform(LossFunction):
    """
    The Kullback-Leibler divergence cost function for uniform distributions.
    """
    name: str = 'KLDivergenceUniform'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.log(expected / actual)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class KLDivergenceWeighted(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function.
    """
    name: str = 'KLDivergenceWeighted'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(expected * np.log(expected / actual)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class KLDivergenceWeightedBernoulli(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function for Bernoulli distributions.
    """
    name: str = 'KLDivergenceWeightedBernoulli'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(expected * np.log(expected / actual) + (1 - expected) * np.log((1 - expected) / (1 - actual))) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual + (1 - expected) / (1 - actual) / len(actual)

class KLDivergenceWeightedCategorical(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function for categorical distributions.
    """
    name: str = 'KLDivergenceWeightedCategorical'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(expected * np.log(expected / actual)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class KLDivergenceWeightedGaussian(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function for Gaussian distributions.
    """
    name: str = 'KLDivergenceWeightedGaussian'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.log(expected / actual) + (np.square(actual) + np.square(expected)) / 2) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class KLDivergenceWeightedMultinomial(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function for multinomial distributions.
    """
    name: str = 'KLDivergenceWeightedMultinomial'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(expected * np.log(expected / actual)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class KLDivergenceWeightedPoisson(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function for Poisson distributions.
    """
    name: str = 'KLDivergenceWeightedPoisson'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(expected * np.log(expected / actual) - expected + actual) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual + 1 / len(actual)

class KLDivergenceWeightedUniform(LossFunction):
    """
    The weighted Kullback-Leibler divergence cost function for uniform distributions.
    """
    name: str = 'KLDivergenceWeightedUniform'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.log(expected / actual)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class MeanAbsoluteError(LossFunction):
    """
    The mean absolute error cost function.
    """
    name: str = 'MeanAbsoluteError'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.abs(actual - expected)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return np.sign(actual - expected) / len(actual)

class MeanAbsolutePercentageError(LossFunction):
    """
    The mean absolute percentage error cost function.
    """
    name: str = 'MeanAbsolutePercentageError'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.abs(actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return np.sign(actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps) / len(actual)

class MeanAbsoluteLogarithmicError(LossFunction):
    """
    The mean absolute logarithmic error cost function.
    """
    name: str = 'MeanAbsoluteLogarithmicError'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.abs(np.log(actual + 1) - np.log(expected + 1))) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return np.sign(np.log(actual + 1) - np.log(expected + 1)) / (actual + 1) / len(actual)

class MeanSquaredLogarithmicError(LossFunction):
    """
    The mean squared logarithmic error cost function.
    """
    name: str = 'MeanSquaredLogarithmicError'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.square(np.log(actual + 1) - np.log(expected + 1))) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return 2 * (np.log(actual + 1) - np.log(expected + 1)) / (actual + 1) / len(actual)

class MeanSquaredPercentageError(LossFunction):
    """
    The mean squared percentage error cost function.
    """
    name: str = 'MeanSquaredPercentageError'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.square((actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps))) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return 2 * (actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps) / len(actual)

class MeanSquaredError(LossFunction):
    """
    The mean squared error cost function.
    """
    name: str = 'MeanSquaredError'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.square(actual - expected)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return 2 * (actual - expected) / len(actual)

class MeanSquaredLogarithmicError(LossFunction):
    """
    The mean squared logarithmic error cost function.
    """
    name: str = 'MeanSquaredLogarithmicError'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.square(np.log(actual + 1) - np.log(expected + 1))) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return 2 * (np.log(actual + 1) - np.log(expected + 1)) / (actual + 1) / len(actual)

class MeanSquaredPercentageError(LossFunction):
    """
    The mean squared percentage error cost function.
    """
    name: str = 'MeanSquaredPercentageError'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.square((actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps))) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return 2 * (actual - expected) / np.maximum(np.abs(expected), np.finfo(np.float64).eps) / len(actual)

class MeanSquaredError(LossFunction):
    """
    The mean squared error cost function.
    """
    name: str = 'MeanSquaredError'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.square(actual - expected)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return 2 * (actual - expected) / len(actual)

class Poisson(LossFunction):
    """
    The Poisson cost function.
    """
    name: str = 'Poisson'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(actual - expected * np.log(actual)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return 1 - expected / actual / len(actual)

class CosineSimilarity(LossFunction):
    """
    The cosine similarity cost function.
    """
    name: str = 'CosineSimilarity'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(actual * expected) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return expected / len(actual)

class Hinge(LossFunction):
    """
    The hinge cost function.
    """
    name: str = 'Hinge'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.maximum(0, 1 - actual * expected)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / len(actual)

class SquaredHinge(LossFunction):
    """
    The squared hinge cost function.
    """
    name: str = 'SquaredHinge'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.square(np.maximum(0, 1 - actual * expected))) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -2 * expected * np.maximum(0, 1 - actual * expected) / len(actual)

class LogCosh(LossFunction):
    """
    The log cosh cost function.
    """
    name: str = 'LogCosh'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.log(np.cosh(actual - expected))) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return np.tanh(actual - expected) / len(actual)

class Huber(LossFunction):
    """
    The Huber cost function.
    """
    name: str = 'Huber'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.where(np.abs(actual - expected) < 1, 0.5 * np.square(actual - expected), np.abs(actual - expected) - 0.5)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return np.where(np.abs(actual - expected) < 1, actual - expected, np.sign(actual - expected)) / len(actual)

class Log(LossFunction):
    """
    The log cost function.
    """
    name: str = 'Log'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.log(1 + np.exp(-actual * expected))) / len(actual)


    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
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

    def __call__(self, actual: np.array, expected: np.array) -> float:
        return np.sum(np.where(actual >= expected, self.q * (actual - expected), (1 - self.q) * (expected - actual))) / len(actual)

    def derivative(self, actual: np.array, expected: np.array) -> np.array:
        return np.where(actual >= expected, self.q, 1 - self.q) / len(actual)

class LogPoisson(LossFunction):
    """
    The log Poisson cost function.
    """
    name: str = 'LogPoisson'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.exp(actual) - actual * expected) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return np.exp(actual) - expected / len(actual)

class KLDivergenceGaussian(LossFunction):
    """
    The Kullback-Leibler divergence cost function for Gaussian distributions.
    """
    name: str = 'KLDivergenceGaussian'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.log(expected / actual) + (np.square(actual) + np.square(expected)) / 2) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual / len(actual)

class PoissonNLL(LossFunction):
    """
    The Poisson negative log likelihood cost function.
    """
    name: str = 'PoissonNLL'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(actual - expected * np.log(actual)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return 1 - expected / actual / len(actual)

class KLDivergenceBernoulli(LossFunction):
    """
    The Kullback-Leibler divergence cost function for Bernoulli distributions.
    """
    name: str = 'KLDivergenceBernoulli'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(expected * np.log(expected / actual) + (1 - expected) * np.log((1 - expected) / (1 - actual))) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / actual + (1 - expected) / (1 - actual) / len(actual)

class CategoricalHinge(LossFunction):
    """
    The categorical hinge cost function.
    """
    name: str = 'CategoricalHinge'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.maximum(0, 1 - actual * expected)) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return -expected / len(actual)

class SquaredLog(LossFunction):
    """
    The squared log cost function.
    """
    name: str = 'SquaredLog'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.square(np.log(actual + 1) - np.log(expected + 1))) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return 2 * (np.log(actual + 1) - np.log(expected + 1)) / (actual + 1) / len(actual)

class PoissonNLLLog(LossFunction):
    """
    The Poisson negative log likelihood cost function.
    """
    name: str = 'PoissonNLLLog'

    @classmethod
    def __call__(cls, actual: np.array, expected: np.array) -> float:
        return np.sum(np.exp(actual) - actual * expected) / len(actual)

    @classmethod
    def derivative(cls, actual: np.array, expected: np.array) -> np.array:
        return np.exp(actual) - expected / len(actual)
