from typing import Tuple

from websearchclassifier.config.baseline.base import BaselineConfig
from websearchclassifier.config.baseline.type import BaselineType


class TFIDFConfig(BaselineConfig):
    type: BaselineType = BaselineType.TFIDF
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_document_frequency: int = 2
    max_document_frequency: float = 0.95
