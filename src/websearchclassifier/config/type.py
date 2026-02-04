from typing import Type

from websearchclassifier.config.baseline.implementations.ftext import FastTextConfig
from websearchclassifier.config.baseline.implementations.herbert import HerBERTConfig
from websearchclassifier.config.baseline.implementations.tfidf import TFIDFConfig
from websearchclassifier.config.baseline.type import BaselineType
from websearchclassifier.config.classifier.implementations.logistic import LogisticRegressionConfig
from websearchclassifier.config.classifier.implementations.mlp import MLPConfig
from websearchclassifier.config.classifier.implementations.svm import SVMConfig
from websearchclassifier.config.classifier.type import ClassifierType
from websearchclassifier.config.model.search import WebSearchClassifierConfig
from websearchclassifier.config.types import BaselineConfigUnion, ClassifierConfigUnion
from websearchclassifier.utils import Kwargs


def get_baseline_config_class(baseline_type: BaselineType) -> Type[BaselineConfigUnion]:
    """
    Factory function to get the appropriate BaselineConfig class based on the model type.

    Args:
        baseline_type (ModelType): The type of the model for which to get the configuration class.

    Returns:
        Type[BaselineConfigUnion]: The corresponding BaselineConfig class.

    Raises:
        KeyError: If the model type is unsupported.
    """
    match baseline_type:
        case BaselineType.TFIDF:
            return TFIDFConfig

        case BaselineType.FASTTEXT:
            return FastTextConfig

        case BaselineType.HERBERT:
            return HerBERTConfig

    raise KeyError(f"Unsupported model type: {baseline_type}")


def get_classifier_config_class(classifier_type: ClassifierType) -> Type[ClassifierConfigUnion]:
    """
    Factory function to get the appropriate ClassifierConfig class based on the classifier type.

    Args:
        classifier_type (ClassifierType): The type of the classifier for which to get the configuration class.

    Returns:
        Type[ClassifierConfigUnion]: The corresponding ClassifierConfig class.

    Raises:
        KeyError: If the classifier type is unsupported.
    """
    match classifier_type:
        case ClassifierType.LOGISTIC_REGRESSION:
            return LogisticRegressionConfig

        case ClassifierType.MLP:
            return MLPConfig

        case ClassifierType.SVM:
            return SVMConfig

    raise KeyError(f"Unsupported classifier type: {classifier_type}")


def load_model_config(
    baseline_type: BaselineType,
    classifier_type: ClassifierType,
    baseline_config_dict: Kwargs,
    classifier_config_dict: Kwargs,
) -> WebSearchClassifierConfig:
    """
    Load model and classifier configurations based on their types and provided configuration dictionaries.

    Args:
        baseline_type (BaselineType): The type of the baseline model.
        classifier_model_type (ClassifierType): The type of the classifier model.
        baseline_config_dict (Kwargs): Configuration parameters for the baseline model.
        classifier_config_dict (Kwargs): Configuration parameters for the classifier model.
    """
    BaselineConfigClass = get_baseline_config_class(baseline_type)
    ClassifierConfigClass = get_classifier_config_class(classifier_type)
    baseline_config: BaselineConfigUnion = BaselineConfigClass(**baseline_config_dict)
    classifier_config: ClassifierConfigUnion = ClassifierConfigClass(**classifier_config_dict)
    return WebSearchClassifierConfig(
        baseline=baseline_config,
        classifier=classifier_config,
    )
