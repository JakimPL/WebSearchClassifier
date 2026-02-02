from websearchclassifier.config.model.base import SearchClassifierConfig
from websearchclassifier.config.model.implementations.ftext import FastTextSearchClassifierConfig
from websearchclassifier.config.model.implementations.tfidf import TfidfSearchClassifierConfig

__all__ = [
    "SearchClassifierConfig",
    "FastTextSearchClassifierConfig",
    "TfidfSearchClassifierConfig",
]
