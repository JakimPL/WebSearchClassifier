from typing import TypeAlias, Union

from websearchclassifier.config.baseline.implementations.ftext import FastTextConfig
from websearchclassifier.config.baseline.implementations.herbert import HerBERTConfig
from websearchclassifier.config.baseline.implementations.tfidf import TFIDFConfig
from websearchclassifier.config.classifier.implementations.logistic import LogisticRegressionConfig
from websearchclassifier.config.classifier.implementations.mlp import MLPConfig
from websearchclassifier.config.classifier.implementations.svm import SVMConfig

BaselineConfigUnion: TypeAlias = Union[
    TFIDFConfig,
    FastTextConfig,
    HerBERTConfig,
]
ClassifierConfigUnion: TypeAlias = Union[
    LogisticRegressionConfig,
    MLPConfig,
    SVMConfig,
]
