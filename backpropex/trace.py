"""
This module provides tracing functions for the backpropex package.

Adding a trace to a `Network`, `Trainer` or `Netgraph` will allow you to
trace the results of the steps without the overhead of creating a graph
image for each step.
"""

from typing import Any, Optional
from backpropex.protocols import Filter, Trace
from backpropex.steps import StepResultAny, StepType
from backpropex.utils import make

class BaseTrace(Trace):
    """
    A base class for traces.
    """
    _filter: Optional[Filter] = None


    def __init__(self,
                 filter: Optional[Filter|type[Filter]] = None,
                 **kwargs: Any) -> None:
        """
        Initialize the trace.

        :param kwargs: Arguments for the trace.
        """
        if filter is not None:
            self._filter = make(filter, Filter)
        super().__init__(**kwargs)

    def filter(self, step: StepType, result: StepResultAny) -> bool:
        if self._filter is None:
            return True
        return self._filter(step, result)

    def __call__(self, step: StepType, result: StepResultAny, /,
                 **kwargs: Any) -> None:
        ...


class PrintTrace(BaseTrace):
    """
    A trace that prints the results.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __call__(self, step: StepType, result: StepResultAny, /,
                 **kwargs: Any) -> None:
        """
        Trace a step result.

        :param step: The step type.
        :param result: The step result.
        """
        if self.filter(step, result):
            print(f'{step}: {result}')

class CollectTrace(BaseTrace):
    """
    A trace that collects the results.
    """
    results: list[StepResultAny]
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.results: list[StepResultAny] = []

    def __call__(self, step: StepType, result: StepResultAny, /,
                 **kwargs: Any) -> None:
        """
        Trace a step result.

        :param step: The step type.
        :param result: The step result.
        """
        if self.filter(step, result):
            self.results.append(result)

class NullTrace(Trace):
    """
    A trace that does nothing.
    """
    def __call__(self, step: StepType, result: StepResultAny, /,
                 **kwargs: Any) -> None:
        """
        Trace a step result.

        :param step: The step type.
        :param result: The step result.
        """
        pass

__all__ = [
    'BaseTrace',
    'CollectTrace',
    'NullTrace',
    'PrintTrace',
]
