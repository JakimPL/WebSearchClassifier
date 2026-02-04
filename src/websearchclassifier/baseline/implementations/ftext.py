from __future__ import annotations

from pathlib import Path

import fasttext
import numpy as np
import numpy.typing as npt
from fasttext.FastText import _FastText

from websearchclassifier.baseline.base import Baseline
from websearchclassifier.config import FastTextConfig
from websearchclassifier.utils import logger


class FastTextBaseline(Baseline[_FastText]):
    config: FastTextConfig

    def __init__(self, config: FastTextConfig) -> None:
        if not isinstance(config, FastTextConfig):
            raise TypeError(f"Expected FastTextBaselineConfig, got {type(config)}")

        super().__init__(config)

        path = Path(config.embeddings_path)
        if not path.exists():
            raise FileNotFoundError(
                f"FastText model not found at: {path}\n"
                "Download Polish model: wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz"
            )

        logger.info("Loading FastText embeddings from %s...", path)
        self.model: _FastText = fasttext.load_model(str(path))
        self._is_fitted = True
        logger.info("Loaded embeddings (dim=%d)", self.model.get_dimension())

    def fit(self, prompts: npt.NDArray[np.str_]) -> None:
        pass

    def transform(self, prompts: npt.NDArray[np.str_]) -> npt.NDArray[np.floating]:
        embedding = np.zeros((len(prompts), self.embedding_dim), dtype=np.float32)
        lowered_prompts = np.strings.lower(prompts)
        for i, text in enumerate(lowered_prompts):
            words = text.split()
            if words:
                word_vectors = np.array([self.model.get_word_vector(word) for word in words])
                embedding[i] = word_vectors.mean(axis=0)

        return embedding

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def embedding_dim(self) -> int:
        embedding_dim: int = self.model.get_dimension()
        return embedding_dim
