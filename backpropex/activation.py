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

from backpropex.protocols import ActivationFunction

_registry: dict[str, 'ActivationFunction'] = {}
def register(af: 'ActivationFunction'):
    """
    Register an activation function.
    """
    _registry[af.name] = af

def get(name: str) -> 'ActivationFunction|None':
    """
    Get an activation function by name.
    """
    return _registry[name]

def names() -> list[str]:
    """
    Get the names of all activation functions.
    """
    return list(_registry.keys())

def activation(cls: type[ActivationFunction]) -> ActivationFunction:
    """
    Decorator to register an activation function.
    """
    af = cls()
    register(af)
    return af

@activation
class ACT_ReLU:
    """The standard ReLU activation function."""
    name = 'ReLU'

    def __call__(self, x: float) -> float:
        return max(0.0, x)

    def derivative(self, x: float) -> float:
        return 1.0 if x > 0.0 else 0.0

@activation
class ACT_Softmax:
    """The standard softmax activation function."""
    name = 'Softmax'

    def __call__(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def derivative(self, x: float) -> float:
        return self(x) * (1.0 - self(x))

@activation
class ACT_Sigmoid:
    """The standard sigmoid activation function."""
    name = 'Sigmoid'

    def __call__(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def derivative(self, x: float) -> float:
        return self(x) * (1.0 - self(x))

@activation
class ACT_Tanh:
    """The standard tanh activation function."""
    name = 'Tanh'

    def __call__(self, x: float) -> float:
        return math.tanh(x)

    def derivative(self, x: float) -> float:
        return 1.0 - self(x) ** 2.0

@activation
class ACT_Identity:
    """The identity activation function."""
    name = 'Identity'

    def __call__(self, x: float) -> float:
        return x

    def derivative(self, x: float) -> float:
        return 1.0

@activation
class ACT_LeakyReLU:
    """The leaky ReLU activation function."""
    name = 'Leaky ReLU'

    def __call__(self, x: float) -> float:
        return max(0.01 * x, x)

    def derivative(self, x: float) -> float:
        return 1.0 if x > 0.0 else 0.01

@activation
class ACT_ELU:
    """The exponential linear unit activation function."""
    name = 'ELU'

    def __call__(self, x: float) -> float:
        return x if x > 0.0 else 0.01 * (math.exp(x) - 1.0)

    def derivative(self, x: float) -> float:
        return 1.0 if x > 0.0 else 0.01 * math.exp(x)

@activation
class ACT_Softplus:
    """The softplus activation function."""
    name = 'Softplus'

    def __call__(self, x: float) -> float:
        return math.log(1.0 + math.exp(x))

    def derivative(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

@activation
class ACT_Swish:
    """The swish activation function."""
    name = 'Swish'

    def __call__(self, x: float) -> float:
        return x / (1.0 + math.exp(-x))

    def derivative(self, x: float) -> float:
        return self(x) + (1.0 - self(x)) / (1.0 + math.exp(-x))

@activation
class ACT_SiLU:
    """The sigmoid-weighted linear unit activation function."""
    name = 'SiLU'

    def __call__(self, x: float) -> float:
        return x / (1.0 + math.exp(-x))

    def derivative(self, x: float) -> float:
        return self(x) + (1.0 - self(x)) / (1.0 + math.exp(-x))

@activation
class ACT_Softsign:
    """The softsign activation function."""
    name = 'Softsign'

    def __call__(self, x: float) -> float:
        return x / (1.0 + abs(x))

    def derivative(self, x: float) -> float:
        return 1.0 / (1.0 + abs(x)) ** 2.0

@activation
class ACT_SQNL:
    """The square nonlinearity activation function."""
    name = 'SQNL'

    def __call__(self, x: float) -> float:
        return 1.0 if x > 2.0 else -1.0 if x < -2.0 else x - x ** 2.0 / 4.0

    def derivative(self, x: float) -> float:
        return 1.0 if x > 2.0 else -1.0 if x < -2.0 else 1.0 - x / 2.0

@activation
class ACT_BentIdentity:
    """The bent identity activation function."""
    name = 'Bent Identity'

    def __call__(self, x: float) -> float:
        return (math.sqrt(x ** 2.0 + 1.0) - 1.0) / 2.0 + x

    def derivative(self, x: float) -> float:
        return x / (2.0 * math.sqrt(x ** 2.0 + 1.0)) + 1.0

@activation
class ACT_Gaussian:
    """The Gaussian activation function."""
    name = 'Gaussian'

    def __call__(self, x: float) -> float:
        return math.exp(-x ** 2.0)

    def derivative(self, x: float) -> float:
        return -2.0 * x * math.exp(-x ** 2.0)

@activation
class ACT_Sinc:
    """The cardinal sine activation function."""
    name = 'Sinc'

    def __call__(self, x: float) -> float:
        return 1.0 if x == 0.0 else math.sin(x) / x

    def derivative(self, x: float) -> float:
        return 0.0 if x == 0.0 else math.cos(x) / x - math.sin(x) / x ** 2.0

@activation
class ACT_Sinusoid:
    """The sinusoid activation function."""
    name = 'Sinusoid'

    def __call__(self, x: float) -> float:
        return math.sin(x)

    def derivative(self, x: float) -> float:
        return math.cos(x)

@activation
class ACT_SincNet:
    """The SincNet activation function."""
    name = 'SincNet'

    def __call__(self, x: float) -> float:
        return math.sin(x) / x

    def derivative(self, x: float) -> float:
        return math.cos(x) / x - math.sin(x) / x ** 2.0

@activation
class ACT_SoftExponential:
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
        self.name = f'Soft Exponential (alpha={alpha})'
        register(self)

    def __call__(self, x: float) -> float:
        return (math.log(1.0 + math.exp(self.alpha * x)) + self.alpha * x) / self.alpha if self.alpha != 0.0 else x

    def derivative(self, x: float) -> float:
        return math.exp(self.alpha * x) / (1.0 + math.exp(self.alpha * x)) if self.alpha != 0.0 else 1.0

@activation
class ACT_SoftClipping:
    """The soft clipping activation function."""
    name = 'Soft Clipping'

    def __call__(self, x: float) -> float:
        return 0.5 * (x + abs(x)) / (1.0 + abs(x))

    def derivative(self, x: float) -> float:
        return 0.5 / (1.0 + abs(x)) ** 2.0

@activation
class ACT_SoftShrink:
    """The soft shrinkage activation function."""
    name = 'Soft Shrink'

    def __call__(self, x: float) -> float:
        return x - 1.0 if x > 1.0 else x + 1.0 if x < -1.0 else 0.0

    def derivative(self, x: float) -> float:
        return 1.0 if x > 1.0 else 1.0 if x < -1.0 else 0.0

@activation
class ACT_Softmin:
    """The softmin activation function."""
    name = 'Softmin'

    def __call__(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def derivative(self, x: float) -> float:
        return self(x) * (1.0 - self(x))

__all__ = [
    'ACT_ReLU',
    'ACT_Softmax',
    'ACT_Sigmoid',
    'ACT_Tanh',
    'ACT_Identity',
    'ACT_LeakyReLU',
    'ACT_ELU',
    'ACT_Softplus',
    'ACT_Swish',
    'ACT_SiLU',
    'ACT_Softsign',
    'ACT_SQNL',
    'ACT_BentIdentity',
    'ACT_Gaussian',
    'ACT_Sinc',
    'ACT_Sinusoid',
    'ACT_SincNet',
    'ACT_SoftExponential',
    'ACT_SoftClipping',
    'ACT_SoftShrink',
    'ACT_Softmin',
    'register',
    'get',
    'names',
    'activation',
]
