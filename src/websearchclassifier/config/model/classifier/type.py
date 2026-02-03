from __future__ import annotations

from enum import Enum
from typing import Literal, TypeAlias, Union


class ClassifierType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"


ClassifierTypeLiteral = Literal["logistic_regression", "svm"]
ClassifierTypeLike: TypeAlias = Union[ClassifierTypeLiteral, ClassifierType]
