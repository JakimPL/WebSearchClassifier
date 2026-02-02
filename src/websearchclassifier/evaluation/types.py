from typing import Any, Callable, Protocol, TypeAlias, Union

from websearchclassifier.dataset import Labels, Predictions
from websearchclassifier.utils import Float

InternalMetric = Callable[[Labels, Predictions], float]


class SklearnMetric(Protocol):
    def __call__(
        self,
        y_true: Any,
        y_pred: Any,
        **kwargs: object,
    ) -> Float: ...


Metric: TypeAlias = Union[InternalMetric, SklearnMetric]
