from __future__ import annotations

from typing import Any, Optional, Self, TypeAlias, Union

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

from websearchclassifier.config import HerBERTSearchClassifierConfig
from websearchclassifier.dataset import Dataset, Prompts
from websearchclassifier.model.base import SearchClassifier
from websearchclassifier.utils import Device, Pathlike, String, Weights, logger

Backend: TypeAlias = Union[TokenizersBackend, SentencePieceBackend]


class HerBERTSearchClassifier(SearchClassifier[HerBERTSearchClassifierConfig]):
    def __init__(self, config: HerBERTSearchClassifierConfig):
        if not isinstance(config, HerBERTSearchClassifierConfig):
            raise TypeError(f"Expected HerBERTSearchClassifierConfig, got {type(config)}")

        super().__init__(config)

        self.device: Device = Device.resolve(config.device)
        self.tokenizer: Optional[Backend] = None
        self.model: Optional[PreTrainedModel] = None

    @property
    def is_model_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _load_model(self) -> None:
        if self.is_model_loaded:
            return

        logger.info("Loading HerBERT model: %s", self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name)
        assert self.model is not None
        assert self.tokenizer is not None

        self.device.to(self.model)
        self.model.eval()
        logger.info("Model loaded on device: %s", self.device)

    def _encode_inputs(self, inputs: BatchEncoding) -> npt.NDArray[np.floating]:
        """
        Encode tokenized inputs through the model.

        Args:
            inputs: Tokenized inputs from the tokenizer

        Returns:
            Array of embeddings, shape (batch_size, hidden_size)
        """
        self._check_model_loaded()
        assert self.model is not None

        inputs_on_device = self.device.to(inputs)
        with torch.no_grad():
            outputs = self.model(**inputs_on_device)
            embeddings: npt.NDArray[np.floating] = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def _encode_text(self, text: String) -> npt.NDArray[np.floating]:
        embedding: npt.NDArray[np.floating] = self._encode_batch(np.array([text]))[0]
        return embedding

    def _encode_batch(self, texts: npt.NDArray[np.str_]) -> npt.NDArray[np.floating]:
        self._check_model_loaded()
        assert self.tokenizer is not None

        num_batches = int(np.ceil(len(texts) / self.config.batch_size))
        batches = np.array_split(texts, num_batches)

        all_embeddings = []
        for batch in batches:
            inputs: BatchEncoding = self.tokenizer(
                batch.tolist(),
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            embeddings = self._encode_inputs(inputs)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def train(self, dataset: Dataset, weights: Weights) -> Self:
        self._load_model()

        logger.info("Encoding %s prompts with HerBERT...", len(dataset.prompts))
        features = self._encode_batch(dataset.prompts)
        return self._train(features, dataset.labels, weights)

    def predict(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.bool_]:
        features = self._get_features(prompts)
        predictions: np.ndarray = self.classifier.predict(features)
        return predictions.astype(np.bool_)

    def predict_proba(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.floating]:
        features = self._get_features(prompts)
        proba: npt.NDArray[np.floating] = self.classifier.predict_proba(features)
        return proba

    def save(self, filepath: Pathlike) -> None:
        """
        Save the classifier to a file.

        The baseline HerBERT model is NOT saved and will be reloaded from HuggingFace

        Args:
            filepath: Path to save the classifier (e.g., 'model.pkl').
        """
        self._save_state(filepath)
        logger.info("HerBERT classifier saved to: '%s'", filepath)
        logger.warning("HerBERT model NOT included. Will be reloaded from HuggingFace on load()")

    @classmethod
    def load(cls, filepath: Pathlike, **kwargs: Any) -> Self:
        """
        Load a trained HerBERTSearchClassifier from file.

        Args:
            filepath: Path to the saved classifier.

        Returns:
            Loaded HerBERTSearchClassifier instance.
        """
        model = cls._load_state(filepath)
        model._load_model()

        logger.info("HerBERT classifier loaded from: '%s'", filepath)
        return model

    def _get_features(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.floating]:
        self.check_is_fitted()
        array: npt.NDArray[np.str_] = self.normalize_prompts(prompts)
        return self._encode_batch(array)

    def _check_model_loaded(self) -> None:
        if not self.is_model_loaded:
            raise RuntimeError("HerBERT model not loaded. Call _load_model() first.")

    def __repr__(self) -> str:
        classifier_type = type(self.classifier).__name__
        return (
            f"HerBERTSearchClassifier("
            f"classifier={classifier_type}, "
            f"device={self.device}, "
            f"batch_size={self.config.batch_size}, "
            f"fitted={self._is_fitted}"
            f")"
        )
