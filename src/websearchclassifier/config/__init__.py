from websearchclassifier.config.base import SearchClassifierConfig
from websearchclassifier.config.dataset import DatasetConfig
from websearchclassifier.config.implementations.ftext import FastTextSearchClassifierConfig
from websearchclassifier.config.implementations.tfidf import TfidfSearchClassifierConfig

__all__ = [
    "DatasetConfig",
    "SearchClassifierConfig",
    "TfidfSearchClassifierConfig",
    "FastTextSearchClassifierConfig",
]
