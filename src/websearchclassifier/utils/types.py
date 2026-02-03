from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Protocol, TypeAlias, TypeVar, Union, get_args

import numpy as np
import torch
from sklearn.base import ClassifierMixin

TorchModuleT = TypeVar("TorchModuleT", bound=torch.nn.Module)
ClassifierT_co = TypeVar("ClassifierT_co", bound=ClassifierMixin, covariant=True)

Bool: TypeAlias = Union[bool, np.bool_]
Integer: TypeAlias = Union[int, np.integer]
Float: TypeAlias = Union[float, np.floating]
String: TypeAlias = Union[str, np.str_]
Pathlike: TypeAlias = Union[str, Path]

Weights: TypeAlias = Dict[Integer, Float]


def does_belong_to_union(value: Any, union_type: Any) -> bool:
    return isinstance(value, get_args(union_type))


def is_bool(value: Any) -> bool:
    return does_belong_to_union(value, Bool)


def is_integer(value: Any) -> bool:
    return does_belong_to_union(value, Integer)


def is_float(value: Any) -> bool:
    return does_belong_to_union(value, Float)


def is_string(value: Any) -> bool:
    return does_belong_to_union(value, String)


class ProbabilisticClassifier(Protocol[ClassifierT_co]):
    """
    Protocol for classifiers that provide probability estimates.
    """

    coef_: np.ndarray

    def fit(self, X: Any, y: Any) -> ClassifierT_co: ...  # pylint: disable=invalid-name

    def predict(self, X: Any) -> np.ndarray: ...  # pylint: disable=invalid-name

    def predict_proba(self, X: Any) -> np.ndarray: ...  # pylint: disable=invalid-name


ProbabilisticClassifierT_co = TypeVar(
    "ProbabilisticClassifierT_co",
    bound=ProbabilisticClassifier[Any],
    covariant=True,
)
