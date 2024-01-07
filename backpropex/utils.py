"""
Implementation utilities for backpropex.
"""

from typing import Optional, cast

def _ids():
    """Generate unique ids."""
    idx = 0
    while True:
        yield idx
        idx += 1

ids = _ids()

def make[T](obj: T|type[T], t: Optional[type[T]] = None) -> T:
    """
    Make an instance if given a type.
    Otherwise, retur the instance.
    """
    if obj is None:
        raise TypeError('obj must not be None')
    if t is not None:
        val = make(obj)
        if isinstance(val, t):
            return cast(T, val)
        else:
            raise TypeError(f'Expected {t}, got {type(val)}')

    if isinstance(obj, type):
        return cast(T, obj())
    else:
        return cast(T, obj)

__all__ = [
    '_ids', 'ids', 'make',
]
