from typing import Any, Protocol, TypeAlias, Union

import numpy as np

from websearchclassifier.dataset import Labels, Predictions
from websearchclassifier.utils import Float


class InternalMetric(Protocol):
    def __call__(
        self,
        labels: Union[np.ndarray, Labels],
        predictions: Union[np.ndarray, Predictions],
    ) -> Float: ...


class SklearnMetric(Protocol):
    def __call__(
        self,
        y_true: Any,
        y_pred: Any,
        **kwargs: object,
    ) -> Float: ...


Metric: TypeAlias = Union[InternalMetric, SklearnMetric]
