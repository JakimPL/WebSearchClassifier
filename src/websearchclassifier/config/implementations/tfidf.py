from typing import Tuple

from websearchclassifier.config.base import SearchClassifierConfig


class TfidfSearchClassifierConfig(SearchClassifierConfig):
    name: str = "tfidf"
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_document_frequency: int = 2
    max_document_frequency: float = 0.95
    regularization_strength: float = 1.0
    random_state: int = 137
