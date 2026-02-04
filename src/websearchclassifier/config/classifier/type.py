from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypeAlias, Union


class ClassifierType(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    MLP = "mlp"


ClassifierTypeLiteral = Literal["logistic_regression", "svm", "mlp"]
ClassifierTypeLike: TypeAlias = Union[ClassifierTypeLiteral, ClassifierType]
