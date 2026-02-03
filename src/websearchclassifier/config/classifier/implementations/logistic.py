from typing import Literal

from sklearn.linear_model import LogisticRegression

from websearchclassifier.config.classifier.base import ClassifierConfig
from websearchclassifier.config.classifier.type import ClassifierType


class LogisticRegressionConfig(ClassifierConfig[LogisticRegression]):
    type: ClassifierType = ClassifierType.LOGISTIC_REGRESSION
    regularization_strength: float = 1.0
    max_iterations: int = 1000
    solver: Literal["liblinear", "lbfgs", "newton-cg", "sag", "saga"] = "liblinear"
