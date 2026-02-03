from typing import Literal

from sklearn.svm import SVC

from websearchclassifier.config.model.classifier.base import ClassifierConfig
from websearchclassifier.config.model.classifier.type import ClassifierType


class SVMConfig(ClassifierConfig[SVC]):
    type: ClassifierType = ClassifierType.SVM
    regularization_strength: float = 1.0
    probability: bool = True
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf"
