from typing import List, Literal

from websearchclassifier.config.classifier.base import ClassifierConfig
from websearchclassifier.config.classifier.type import ClassifierType


class MLPConfig(ClassifierConfig):
    type: ClassifierType = ClassifierType.MLP
    hidden_layer_sizes: List[int] = [128, 64]
    activation: Literal["identity", "logistic", "tanh", "relu"] = "relu"
    learning_rate: float = 0.001
    max_iterations: int = 200
    batch_size: int = 32
    dropout: float = 0.3
    early_stopping: bool = True
    validation_fraction: float = 0.1
