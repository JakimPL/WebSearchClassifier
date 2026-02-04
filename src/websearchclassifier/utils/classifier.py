from __future__ import annotations

from typing import Any, Protocol, TypeVar

import numpy as np
import torch
from sklearn.base import ClassifierMixin

ModelT = TypeVar("ModelT")
TorchModuleT = TypeVar("TorchModuleT", bound=torch.nn.Module)
ClassifierT_co = TypeVar("ClassifierT_co", bound=ClassifierMixin, covariant=True)


class ProbabilisticClassifier(Protocol[ClassifierT_co]):
    """
    Protocol for classifiers that provide probability estimates.
    """

    def fit(self, X: Any, y: Any) -> ClassifierT_co: ...  # pylint: disable=invalid-name

    def predict(self, X: Any) -> np.ndarray: ...  # pylint: disable=invalid-name

    def predict_proba(self, X: Any) -> np.ndarray: ...  # pylint: disable=invalid-name

    def set_params(self, **params: Any) -> ClassifierT_co: ...


ProbabilisticClassifierT_co = TypeVar(
    "ProbabilisticClassifierT_co",
    bound=ProbabilisticClassifier[Any],
    covariant=True,
)
