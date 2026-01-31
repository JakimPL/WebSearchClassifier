from websearchclassifier.model.base import SearchClassifier
from websearchclassifier.model.implementations.ftext import FastTextSearchClassifier
from websearchclassifier.model.implementations.tfidf import TfidfSearchClassifier

__all__ = [
    "SearchClassifier",
    "FastTextSearchClassifier",
    "TfidfSearchClassifier",
]
