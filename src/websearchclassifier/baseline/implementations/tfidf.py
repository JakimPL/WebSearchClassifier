from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer

from websearchclassifier.baseline.base import Baseline
from websearchclassifier.config import TFIDFConfig
from websearchclassifier.utils import logger


class TFIDFBaseline(Baseline[TfidfVectorizer]):
    config: TFIDFConfig

    def __init__(self, config: TFIDFConfig) -> None:
        if not isinstance(config, TFIDFConfig):
            raise TypeError(f"Expected TFIDFBaselineConfig, got {type(config)}")

        super().__init__(config)

        self.model = TfidfVectorizer(
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            min_df=config.min_document_frequency,
            max_df=config.max_document_frequency,
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b\w+\b",
        )

        logger.info("Initialized TF-IDF baseline")

    def fit(self, prompts: npt.NDArray[np.str_]) -> None:
        self.model.fit(prompts)
        self._is_fitted = True

    def transform(self, prompts: npt.NDArray[np.str_]) -> npt.NDArray[np.floating]:
        if not self._is_fitted:
            raise RuntimeError("TF-IDF baseline not fitted")

        sparse_result = self.model.transform(prompts)
        return np.array(sparse_result.todense())

    @property
    def is_trainable(self) -> bool:
        return True

    @property
    def embedding_dim(self) -> int:
        return len(self.model.vocabulary_)
