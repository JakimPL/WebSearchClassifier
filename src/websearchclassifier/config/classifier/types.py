from typing import TypeAlias, Union

from websearchclassifier.config.classifier.implementations.logistic import LogisticRegressionConfig
from websearchclassifier.config.classifier.implementations.mlp import MLPConfig
from websearchclassifier.config.classifier.implementations.svm import SVMConfig

ClassifierConfigUnion: TypeAlias = Union[
    LogisticRegressionConfig,
    MLPConfig,
    SVMConfig,
]
