from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypeAlias, Union


class ClassifierType(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"


ClassifierTypeLiteral = Literal["logistic_regression", "svm"]
ClassifierTypeLike: TypeAlias = Union[ClassifierTypeLiteral, ClassifierType]
