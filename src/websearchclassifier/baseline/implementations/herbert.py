from __future__ import annotations

from typing import TypeAlias, Union

import numpy as np
import numpy.typing as npt
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    SentencePieceBackend,
    TokenizersBackend,
)

from websearchclassifier.baseline.base import Baseline
from websearchclassifier.config import HerBERTConfig
from websearchclassifier.utils import Device, logger

Backend: TypeAlias = Union[TokenizersBackend, SentencePieceBackend]


class HerBERTBaseline(Baseline[PreTrainedModel]):
    config: HerBERTConfig
    tokenizer: Backend

    def __init__(self, config: HerBERTConfig) -> None:
        if not isinstance(config, HerBERTConfig):
            raise TypeError(f"Expected HerBERTBaselineConfig, got {type(config)}")

        super().__init__(config)

        self.device: Device = Device.resolve(config.device)

        logger.info("Loading HerBERT model: %s", config.model_name)
        self.tokenizer: Backend = AutoTokenizer.from_pretrained(config.model_name)
        self.model: PreTrainedModel = AutoModel.from_pretrained(config.model_name)
        self.max_length: int = self.model.config.max_position_embeddings

        self.device.to(self.model)
        self.model.eval()
        self._is_fitted = True
        logger.info("Model loaded on device: %s", self.device)

    def fit(self, prompts: npt.NDArray[np.str_]) -> None:
        pass

    def transform(self, prompts: npt.NDArray[np.str_]) -> npt.NDArray[np.floating]:
        num_batches = int(np.ceil(len(prompts) / self.config.batch_size))
        batches = np.array_split(prompts, num_batches)

        all_embeddings = []
        for batch in batches:
            inputs: BatchEncoding = self.tokenizer(
                batch.tolist(),
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            inputs_on_device = self.device.to(inputs)
            with torch.no_grad():
                outputs = self.model(**inputs_on_device)
                embeddings: npt.NDArray[np.floating] = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def embedding_dim(self) -> int:
        embedding_dim: int = self.model.config.hidden_size
        return embedding_dim
