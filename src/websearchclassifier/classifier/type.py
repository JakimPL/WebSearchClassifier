from typing import Any, Type

from websearchclassifier.classifier.base import Classifier
from websearchclassifier.classifier.implementations.logistic import LogisticRegressionWrapper
from websearchclassifier.classifier.implementations.mlp import MLPWrapper
from websearchclassifier.classifier.implementations.svm import SVMWrapper
from websearchclassifier.config import ClassifierConfig, ClassifierType


def get_classifier_wrapper_class(classifier_type: ClassifierType) -> Type[Classifier[Any]]:
    """
    Factory function to get the classifier wrapper class based on the classifier type.

    Args:
        classifier_type (ClassifierType): The type of the classifier.

    Returns:
        Type[ClassifierWrapper[Any]]: The corresponding classifier wrapper class.
    """
    match classifier_type:
        case ClassifierType.LOGISTIC_REGRESSION:
            return LogisticRegressionWrapper

        case ClassifierType.MLP:
            return MLPWrapper

        case ClassifierType.SVM:
            return SVMWrapper

    raise KeyError(f"Unsupported classifier wrapper type: {classifier_type}")


def load_classifier_wrapper(config: ClassifierConfig) -> Classifier[Any]:
    """
    Load the classifier wrapper based on the provided configuration.

    Args:
        config (ClassifierConfig): The classifier configuration.

    Returns:
        ClassifierWrapper[Any]: The corresponding classifier wrapper instance.
    """
    WrapperClass = get_classifier_wrapper_class(config.type)
    wrapper: Classifier[Any] = WrapperClass(config)
    return wrapper
