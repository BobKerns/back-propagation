"""
    Filtering functions for backpropex.

    Filtering is done in three steps

    1) Prefiltering: The StepResult is passed as None.
    2) Full filtering. The filter receives the StepResult and can decide whether to
       accept it or not.
    3) Postfiltering: The StepResult will be passed as False. If the filter returns False,
       the processing will be stopped.
"""

from collections.abc import Sequence
from typing import Any, cast
from backpropex.protocols import Filter
from backpropex.steps import StepType, FilterArg, StepTypeAny


class OutputOnlyFilter(Filter):
    """
    A filter for backpropex.
    """
    def __call__[T: (StepTypeAny, StepType)](self, step: StepType, result: FilterArg[T], **kwargs: Any) -> bool:
        """
        Filter a step result.

        :param step: The step type.
        :param result: The step result.
        :return: True if the result is accepted, False otherwise.
        """
        return step == StepType.Output

class EveryNFilter(Filter):
    """
    A filter that only accepts every n-th step.
    It may also limit it to a specific step type.
    """
    types: Sequence[StepType]
    n: int
    count: int
    def __init__(self,
                 n: int = 10,
                 step_type: StepType|Sequence[StepType] = StepType.Output,
                 ):
        self.n = n
        self.count = 0
        self.types = [step_type] if isinstance(step_type, StepType) else step_type

    def __call__[T: (StepTypeAny, StepType)](self, step: StepType, result: FilterArg[T], **kwargs: Any) -> bool:
        if len(self.types) > 0 and step not in self.types:
            return False
        self.count += 1
        if self.count >= self.n:
            self.count = 0
            return True
        return False

class FilterNone(Filter):
    """
    A filter that always returns False.
    """
    def __call__[T: (StepTypeAny, StepType)](self, step: StepType, result: FilterArg[T], **kwargs: Any) -> bool:
        return False


class FilterChain(Filter):
    """
    A filter that wraps a filter with another filter
    """
    next: Filter
    filter: Filter
    def __init__(self, filter: Filter, next: Filter):
        self.filter = filter
        self.next = next

    def __call__[T: (StepTypeAny, StepType)](
            self, step: StepType, result:FilterArg[T],
            **kwargs: Any
            ) -> bool:
        a1: bool = cast(bool, # remove spurious Unknown
                        self.filter(
                            step,
                            result,  # type: ignore
                            **kwargs))
        a2: bool = cast(bool, # remove spurious Unknown.
                        self.next(
                            step,
                            result, # type: ignore
                            **kwargs))
        return a1 and a2

__all__ = [
    'EveryNFilter',
    'OutputOnlyFilter',
    'FilterNone',
    'FilterChain',
]
