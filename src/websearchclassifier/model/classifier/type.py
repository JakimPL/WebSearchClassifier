from typing import Any, Type

from websearchclassifier.config import ClassifierConfig, ClassifierType
from websearchclassifier.config.model.classifier.implementations.logistic import LogisticRegressionConfig
from websearchclassifier.config.model.classifier.implementations.svm import SVMConfig
from websearchclassifier.model.classifier.implementations.logistic import LogisticRegressionWrapper
from websearchclassifier.model.classifier.implementations.svm import SVMWrapper
from websearchclassifier.model.classifier.wrapper import ClassifierWrapper
from websearchclassifier.utils import ProbabilisticClassifierT_co


def get_classifier_config_class(classifier_type: ClassifierType) -> Type[ClassifierConfig[Any]]:
    match classifier_type:
        case ClassifierType.LOGISTIC_REGRESSION:
            return LogisticRegressionConfig

        case ClassifierType.SVM:
            return SVMConfig

    raise ValueError(f"Unsupported classifier type: {classifier_type}")


def get_classifier_wrapper_class(classifier_type: ClassifierType) -> Type[ClassifierWrapper[Any]]:
    match classifier_type:
        case ClassifierType.LOGISTIC_REGRESSION:
            return LogisticRegressionWrapper

        case ClassifierType.SVM:
            return SVMWrapper

    raise ValueError(f"Unsupported classifier type: {classifier_type}")


def load_classifier_wrapper(
    config: ClassifierConfig[ProbabilisticClassifierT_co],
) -> ClassifierWrapper[ProbabilisticClassifierT_co]:
    WrapperClass = get_classifier_wrapper_class(config.type)
    wrapper: ClassifierWrapper[ProbabilisticClassifierT_co] = WrapperClass(config)
    return wrapper
