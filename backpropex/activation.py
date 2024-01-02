"""
Activation functions and their derivatives.

This module contains the activation functions and their derivatives that are
used by the network. The activation functions are implemented as classes
that implement the `ActivationFunction` protocol.

The activation functions are used by the `Node` class to calculate the
activation of a node and its derivative.

Most of these activation functions were suggested by Copilot. I have not verified
that they are correct, but they are interesting, and I have not spotted any error
in the ones I have examined and used.
"""

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

class ACT_Tanh(ActivationFunction):
    """The standard tanh activation function."""
    name = 'Tanh'

    @classmethod
    def __call__(cls, x: float) -> float:
        return math.tanh(x)

    @classmethod
    def derivative(cls, x: float) -> float:
        return 1.0 - cls(x) ** 2.0

class ACT_Identity(ActivationFunction):
    """The identity activation function."""
    name = 'Identity'

    @classmethod
    def __call__(cls, x: float) -> float:
        return x

    @classmethod
    def derivative(cls, x: float) -> float:
        return 1.0

class ACT_LeakyReLU(ActivationFunction):
    """The leaky ReLU activation function."""
    name = 'Leaky ReLU'

    @classmethod
    def __call__(cls, x: float) -> float:
        return max(0.01 * x, x)

    @classmethod
    def derivative(cls, x: float) -> float:
        return 1.0 if x > 0.0 else 0.01

class ACT_ELU(ActivationFunction):
    """The exponential linear unit activation function."""
    name = 'ELU'

    @classmethod
    def __call__(cls, x: float) -> float:
        return x if x > 0.0 else 0.01 * (math.exp(x) - 1.0)

    @classmethod
    def derivative(cls, x: float) -> float:
        return 1.0 if x > 0.0 else 0.01 * math.exp(x)

class ACT_Softplus(ActivationFunction):
    """The softplus activation function."""
    name = 'Softplus'

    @classmethod
    def __call__(cls, x: float) -> float:
        return math.log(1.0 + math.exp(x))

    @classmethod
    def derivative(cls, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

class ACT_Swish(ActivationFunction):
    """The swish activation function."""
    name = 'Swish'

    @classmethod
    def __call__(cls, x: float) -> float:
        return x / (1.0 + math.exp(-x))

    @classmethod
    def derivative(cls, x: float) -> float:
        return cls(x) + (1.0 - cls(x)) / (1.0 + math.exp(-x))

class ACT_SiLU(ActivationFunction):
    """The sigmoid-weighted linear unit activation function."""
    name = 'SiLU'

    @classmethod
    def __call__(cls, x: float) -> float:
        return x / (1.0 + math.exp(-x))

    @classmethod
    def derivative(cls, x: float) -> float:
        return cls(x) + (1.0 - cls(x)) / (1.0 + math.exp(-x))

class ACT_Softsign(ActivationFunction):
    """The softsign activation function."""
    name = 'Softsign'

    @classmethod
    def __call__(cls, x: float) -> float:
        return x / (1.0 + abs(x))

    @classmethod
    def derivative(cls, x: float) -> float:
        return 1.0 / (1.0 + abs(x)) ** 2.0

class ACT_SQNL(ActivationFunction):
    """The square nonlinearity activation function."""
    name = 'SQNL'

    @classmethod
    def __call__(cls, x: float) -> float:
        return 1.0 if x > 2.0 else -1.0 if x < -2.0 else x - x ** 2.0 / 4.0

    @classmethod
    def derivative(cls, x: float) -> float:
        return 1.0 if x > 2.0 else -1.0 if x < -2.0 else 1.0 - x / 2.0

class ACT_BentIdentity(ActivationFunction):
    """The bent identity activation function."""
    name = 'Bent Identity'

    @classmethod
    def __call__(cls, x: float) -> float:
        return (math.sqrt(x ** 2.0 + 1.0) - 1.0) / 2.0 + x

    @classmethod
    def derivative(cls, x: float) -> float:
        return x / (2.0 * math.sqrt(x ** 2.0 + 1.0)) + 1.0

class ACT_Gaussian(ActivationFunction):
    """The Gaussian activation function."""
    name = 'Gaussian'

    @classmethod
    def __call__(cls, x: float) -> float:
        return math.exp(-x ** 2.0)

    @classmethod
    def derivative(cls, x: float) -> float:
        return -2.0 * x * math.exp(-x ** 2.0)

class ACT_Sinc(ActivationFunction):
    """The cardinal sine activation function."""
    name = 'Sinc'

    @classmethod
    def __call__(cls, x: float) -> float:
        return 1.0 if x == 0.0 else math.sin(x) / x

    @classmethod
    def derivative(cls, x: float) -> float:
        return 0.0 if x == 0.0 else math.cos(x) / x - math.sin(x) / x ** 2.0

class ACT_Sinusoid(ActivationFunction):
    """The sinusoid activation function."""
    name = 'Sinusoid'

    @classmethod
    def __call__(cls, x: float) -> float:
        return math.sin(x)

    @classmethod
    def derivative(cls, x: float) -> float:
        return math.cos(x)

class ACT_SincNet(ActivationFunction):
    """The SincNet activation function."""
    name = 'SincNet'

    @classmethod
    def __call__(cls, x: float) -> float:
        return math.sin(x) / x

    @classmethod
    def derivative(cls, x: float) -> float:
        return math.cos(x) / x - math.sin(x) / x ** 2.0

class ACT_SoftExponential(ActivationFunction):
    """
    The soft exponential activation function.

    The soft exponential activation function is a generalization of the ReLU
    activation function. It has an additional parameter, alpha, that controls
    the shape of the function. The ReLU activation function is obtained by
    setting alpha to zero.

    The soft exponential activation function is defined as:

            f(x) = (log(1 + exp(alpha * x)) + alpha * x) / alpha

    The derivative of the soft exponential activation function is:

                f'(x) = exp(alpha * x) / (1 + exp(alpha * x))

    Unlike most activation functions in this module, the soft exponential
    must be instantiated with a value for alpha. The default value is zero,
    which results in the ReLU activation function.
    """
    name = 'Soft Exponential'

    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha

    def __call__(self, x: float) -> float:
        return (math.log(1.0 + math.exp(self.alpha * x)) + self.alpha * x) / self.alpha if self.alpha != 0.0 else x

    def derivative(self, x: float) -> float:
        return math.exp(self.alpha * x) / (1.0 + math.exp(self.alpha * x)) if self.alpha != 0.0 else 1.0

class ACT_SoftClipping(ActivationFunction):
    """The soft clipping activation function."""
    name = 'Soft Clipping'

    @classmethod
    def __call__(cls, x: float) -> float:
        return 0.5 * (x + abs(x)) / (1.0 + abs(x))

    @classmethod
    def derivative(cls, x: float) -> float:
        return 0.5 / (1.0 + abs(x)) ** 2.0

class ACT_SoftShrink(ActivationFunction):
    """The soft shrinkage activation function."""
    name = 'Soft Shrink'

    @classmethod
    def __call__(cls, x: float) -> float:
        return x - 1.0 if x > 1.0 else x + 1.0 if x < -1.0 else 0.0

    @classmethod
    def derivative(cls, x: float) -> float:
        return 1.0 if x > 1.0 else 1.0 if x < -1.0 else 0.0

class ACT_Softmin(ActivationFunction):
    """The softmin activation function."""
    name = 'Softmin'

    @classmethod
    def __call__(cls, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @classmethod
    def derivative(cls, x: float) -> float:
        return cls(x) * (1.0 - cls(x))
