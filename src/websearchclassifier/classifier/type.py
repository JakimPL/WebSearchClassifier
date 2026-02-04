from typing import Any, Type

from websearchclassifier.classifier.implementations.logistic import LogisticRegressionWrapper
from websearchclassifier.classifier.implementations.mlp import MLPWrapper
from websearchclassifier.classifier.implementations.svm import SVMWrapper
from websearchclassifier.classifier.wrapper import ClassifierWrapper
from websearchclassifier.config import ClassifierConfig, ClassifierType, LogisticRegressionConfig, MLPConfig, SVMConfig
from websearchclassifier.utils import ProbabilisticClassifierT_co


def get_classifier_config_class(classifier_type: ClassifierType) -> Type[ClassifierConfig[Any]]:
    """
    Get the classifier configuration class based on the classifier type.

    Args:
        classifier_type (ClassifierType): The type of classifier.

    Returns:
        Type[ClassifierConfig[Any]]: The corresponding classifier configuration class.

    Raises:
        KeyError: If the classifier type is unsupported.
    """
    match classifier_type:
        case ClassifierType.LOGISTIC_REGRESSION:
            return LogisticRegressionConfig

        case ClassifierType.SVM:
            return SVMConfig

        case ClassifierType.MLP:
            return MLPConfig

    raise KeyError(f"Unsupported classifier type: {classifier_type}")


def get_classifier_wrapper_class(classifier_type: ClassifierType) -> Type[ClassifierWrapper[Any]]:
    """
    Get the classifier wrapper class based on the classifier type.

    Args:
        classifier_type (ClassifierType): The type of classifier.

    Returns:
        Type[ClassifierWrapper[Any]]: The corresponding classifier wrapper class.

    Raises:
        KeyError: If the classifier type is unsupported.
    """

    match classifier_type:
        case ClassifierType.LOGISTIC_REGRESSION:
            return LogisticRegressionWrapper

        case ClassifierType.SVM:
            return SVMWrapper

        case ClassifierType.MLP:
            return MLPWrapper

    raise KeyError(f"Unsupported classifier type: {classifier_type}")


def load_classifier_wrapper(
    config: ClassifierConfig[ProbabilisticClassifierT_co],
) -> ClassifierWrapper[ProbabilisticClassifierT_co]:
    """
    Load the classifier wrapper based on the provided configuration.

    Args:
        config (ClassifierConfig[ProbabilisticClassifierT_co]): The classifier configuration.

    Returns:
        ClassifierWrapper[ProbabilisticClassifierT_co]: The corresponding classifier wrapper instance.
    """
    WrapperClass = get_classifier_wrapper_class(config.type)
    wrapper: ClassifierWrapper[ProbabilisticClassifierT_co] = WrapperClass(config)
    return wrapper
