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
from typing import Any, Protocol

class ActivationFunction(Protocol):
    """
    An activation function and its derivative.
    """
    name: str

    _registry: dict[str, 'ActivationFunction'] = {}

    @staticmethod
    def register(cls: 'ActivationFunction') -> None:
        """Register an activation function."""
        ActivationFunction._registry[cls.name] = cls

    @staticmethod
    def names() -> list[str]:
        """Return the names of the registered activation functions."""
        return list(ActivationFunction._registry.keys())

    def __call__(cls, x: float) -> float:
        """Nonlinear activation response of a node."""
        ...

    def derivative(cls, x: float) -> float:
        """Derivative of the activation function."""
        ...
    @classmethod
    def __getattr__(cls, name: str) -> Any:
        """
        Allow accessing loss functions via attribute access.
        """
        return ActivationFunction._registry[name]

    def __init_subclass__(cls) -> None:
        if hasattr(cls, 'name'):
            ActivationFunction.register(cls)
        return super().__init_subclass__()

class ActivationBase(ActivationFunction):
    """Base class to allow defining ActivationFunction objects via class syntax."""

    def __new__(cls, x: float) -> float:
        """Nonlinear activation response of a node."""
        return cls.__call__(cls, x)

    def __init_subclass__(cls) -> None:
        """Wrap the derivative method to allow defining LossFunction objects via class syntax."""
        sub = super().__init_subclass__()
        derivative = cls.derivative
        (sub or cls).derivative = lambda x: derivative(cls, x)
        return sub

class ACT_ReLU(ActivationBase):
    """The standard ReLU activation function."""
    name = 'ReLU'

    def __call__(cls, x: float) -> float:
        return max(0.0, x)

    def derivative(cls, x: float) -> float:
        return 1.0 if x > 0.0 else 0.0

class ACT_Softmax(ActivationBase):
    """The standard softmax activation function."""
    name = 'Softmax'

    def __call__(cls, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def derivative(cls, x: float) -> float:
        return cls(x) * (1.0 - cls(x))

class ACT_Sigmoid(ActivationBase):
    """The standard sigmoid activation function."""
    name = 'Sigmoid'

    def __call__(cls, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def derivative(cls, x: float) -> float:
        return cls(x) * (1.0 - cls(x))

class ACT_Tanh(ActivationBase):
    """The standard tanh activation function."""
    name = 'Tanh'

    def __call__(cls, x: float) -> float:
        return math.tanh(x)

    def derivative(cls, x: float) -> float:
        return 1.0 - cls(x) ** 2.0

class ACT_Identity(ActivationBase):
    """The identity activation function."""
    name = 'Identity'

    def __call__(cls, x: float) -> float:
        return x

    def derivative(cls, x: float) -> float:
        return 1.0

class ACT_LeakyReLU(ActivationBase):
    """The leaky ReLU activation function."""
    name = 'Leaky ReLU'

    def __call__(cls, x: float) -> float:
        return max(0.01 * x, x)

    def derivative(cls, x: float) -> float:
        return 1.0 if x > 0.0 else 0.01

class ACT_ELU(ActivationBase):
    """The exponential linear unit activation function."""
    name = 'ELU'

    def __call__(cls, x: float) -> float:
        return x if x > 0.0 else 0.01 * (math.exp(x) - 1.0)

    def derivative(cls, x: float) -> float:
        return 1.0 if x > 0.0 else 0.01 * math.exp(x)

class ACT_Softplus(ActivationBase):
    """The softplus activation function."""
    name = 'Softplus'

    def __call__(cls, x: float) -> float:
        return math.log(1.0 + math.exp(x))

    def derivative(cls, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

class ACT_Swish(ActivationBase):
    """The swish activation function."""
    name = 'Swish'

    def __call__(cls, x: float) -> float:
        return x / (1.0 + math.exp(-x))

    def derivative(cls, x: float) -> float:
        return cls(x) + (1.0 - cls(x)) / (1.0 + math.exp(-x))

class ACT_SiLU(ActivationBase):
    """The sigmoid-weighted linear unit activation function."""
    name = 'SiLU'

    def __call__(cls, x: float) -> float:
        return x / (1.0 + math.exp(-x))

    def derivative(cls, x: float) -> float:
        return cls(x) + (1.0 - cls(x)) / (1.0 + math.exp(-x))

class ACT_Softsign(ActivationBase):
    """The softsign activation function."""
    name = 'Softsign'

    def __call__(cls, x: float) -> float:
        return x / (1.0 + abs(x))

    def derivative(cls, x: float) -> float:
        return 1.0 / (1.0 + abs(x)) ** 2.0

class ACT_SQNL(ActivationBase):
    """The square nonlinearity activation function."""
    name = 'SQNL'

    def __call__(cls, x: float) -> float:
        return 1.0 if x > 2.0 else -1.0 if x < -2.0 else x - x ** 2.0 / 4.0

    def derivative(cls, x: float) -> float:
        return 1.0 if x > 2.0 else -1.0 if x < -2.0 else 1.0 - x / 2.0

class ACT_BentIdentity(ActivationBase):
    """The bent identity activation function."""
    name = 'Bent Identity'

    def __call__(cls, x: float) -> float:
        return (math.sqrt(x ** 2.0 + 1.0) - 1.0) / 2.0 + x

    def derivative(cls, x: float) -> float:
        return x / (2.0 * math.sqrt(x ** 2.0 + 1.0)) + 1.0

class ACT_Gaussian(ActivationBase):
    """The Gaussian activation function."""
    name = 'Gaussian'

    def __call__(cls, x: float) -> float:
        return math.exp(-x ** 2.0)

    def derivative(cls, x: float) -> float:
        return -2.0 * x * math.exp(-x ** 2.0)

class ACT_Sinc(ActivationBase):
    """The cardinal sine activation function."""
    name = 'Sinc'

    def __call__(cls, x: float) -> float:
        return 1.0 if x == 0.0 else math.sin(x) / x

    def derivative(cls, x: float) -> float:
        return 0.0 if x == 0.0 else math.cos(x) / x - math.sin(x) / x ** 2.0

class ACT_Sinusoid(ActivationBase):
    """The sinusoid activation function."""
    name = 'Sinusoid'

    def __call__(cls, x: float) -> float:
        return math.sin(x)

    def derivative(cls, x: float) -> float:
        return math.cos(x)

class ACT_SincNet(ActivationBase):
    """The SincNet activation function."""
    name = 'SincNet'

    def __call__(cls, x: float) -> float:
        return math.sin(x) / x

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

class ACT_SoftClipping(ActivationBase):
    """The soft clipping activation function."""
    name = 'Soft Clipping'

    def __call__(cls, x: float) -> float:
        return 0.5 * (x + abs(x)) / (1.0 + abs(x))

    def derivative(cls, x: float) -> float:
        return 0.5 / (1.0 + abs(x)) ** 2.0

class ACT_SoftShrink(ActivationBase):
    """The soft shrinkage activation function."""
    name = 'Soft Shrink'

    def __call__(cls, x: float) -> float:
        return x - 1.0 if x > 1.0 else x + 1.0 if x < -1.0 else 0.0

    def derivative(cls, x: float) -> float:
        return 1.0 if x > 1.0 else 1.0 if x < -1.0 else 0.0

class ACT_Softmin(ActivationBase):
    """The softmin activation function."""
    name = 'Softmin'

    def __call__(cls, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def derivative(cls, x: float) -> float:
        return cls(x) * (1.0 - cls(x))
