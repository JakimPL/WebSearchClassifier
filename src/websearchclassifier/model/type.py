from typing import Any, Dict, Type

from websearchclassifier.config import (
    ClassifierType,
    ConfigT,
    FastTextSearchClassifierConfig,
    HerBERTSearchClassifierConfig,
    ModelType,
    SearchClassifierConfig,
    TFIDFSearchClassifierConfig,
)
from websearchclassifier.model.base import SearchClassifier
from websearchclassifier.model.implementations.ftext import FastTextSearchClassifier
from websearchclassifier.model.implementations.herbert import HerBERTSearchClassifier
from websearchclassifier.model.implementations.tfidf import TFIDFSearchClassifier


def get_model_config_class(model_type: ModelType) -> Type[SearchClassifierConfig]:
    match model_type:
        case ModelType.TFIDF:
            return TFIDFSearchClassifierConfig
        case ModelType.FASTTEXT:
            return FastTextSearchClassifierConfig
        case ModelType.HERBERT:
            return HerBERTSearchClassifierConfig

    raise ValueError(f"Unsupported model type: {model_type}")


def get_model_class(model_type: ModelType) -> Type[SearchClassifier[Any]]:
    match model_type:
        case ModelType.TFIDF:
            return TFIDFSearchClassifier
        case ModelType.FASTTEXT:
            return FastTextSearchClassifier
        case ModelType.HERBERT:
            return HerBERTSearchClassifier

    raise ValueError(f"Unsupported model type: {model_type}")


def load_model_config(
    model_type: ModelType,
    classifier_model_type: ClassifierType,
    model_config_dict: Dict[str, Any],
    classifier_config_dict: Dict[str, Any],
) -> SearchClassifierConfig:
    ConfigClass = get_model_config_class(model_type)
    config: SearchClassifierConfig = ConfigClass(
        **model_config_dict,
        classifier_config=classifier_config_dict[classifier_model_type],
    )
    return config


def load_model_from_config(config: ConfigT) -> SearchClassifier[ConfigT]:
    ModelClass = get_model_class(config.type)
    model: SearchClassifier[ConfigT] = ModelClass(config)
    return model
