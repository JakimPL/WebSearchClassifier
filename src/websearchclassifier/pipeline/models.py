from enum import StrEnum
from typing import Any, Type

from websearchclassifier.config import (
    FastTextSearchClassifierConfig,
    SearchClassifierConfig,
    TfidfSearchClassifierConfig,
)
from websearchclassifier.model import FastTextSearchClassifier, SearchClassifier, TfidfSearchClassifier


class ModelType(StrEnum):
    TFIDF = "tfidf"
    FASTTEXT = "fasttext"

    @property
    def config_class(self) -> Type[SearchClassifierConfig]:
        match self:
            case ModelType.TFIDF:
                return TfidfSearchClassifierConfig
            case ModelType.FASTTEXT:
                return FastTextSearchClassifierConfig

        raise ValueError(f"Unsupported model type: {self}")

    @property
    def model_class(self) -> Type[SearchClassifier[Any]]:
        match self:
            case ModelType.TFIDF:
                return TfidfSearchClassifier
            case ModelType.FASTTEXT:
                return FastTextSearchClassifier

        raise ValueError(f"Unsupported model type: {self}")
