from typing import Tuple

from websearchclassifier.config.model.base import SearchClassifierConfig
from websearchclassifier.config.model.type import ModelType


class TFIDFSearchClassifierConfig(SearchClassifierConfig):
    type: ModelType = ModelType.TFIDF
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_document_frequency: int = 2
    max_document_frequency: float = 0.95
