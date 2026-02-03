from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional, Self, Union

import fasttext
import numpy as np
import numpy.typing as npt
from fasttext.FastText import _FastText

from websearchclassifier.config import FastTextSearchClassifierConfig
from websearchclassifier.dataset import Dataset, Prompts
from websearchclassifier.model.base import SearchClassifier
from websearchclassifier.utils import Pathlike, Weights, logger


class FastTextSearchClassifier(SearchClassifier[FastTextSearchClassifierConfig]):
    """
    Binary classifier using FastText embeddings for Polish prompts.

    Uses pre-trained Polish FastText word embeddings with averaging strategy
    to create sentence representations, then trains a classifier on top.

    Better than TF-IDF for semantic understanding but requires downloading
    pre-trained embeddings (~1-6GB depending on version).

    Example:
        >>> classifier = FastTextSearchClassifier()
        >>> classifier.load_embeddings("cc.pl.300.bin")
        >>>
        >>> train_data = [
        ...     ("jaka jest dzisiaj pogoda w Warszawie", True),
        ...     ("oblicz pierwiastek z 144", False),
        ... ]
        >>> prompts, labels = zip(*train_data)
        >>> classifier.fit(prompts, labels)
        >>>
        >>> classifier.predict(["jaka jest aktualna cena bitcoina"])
    """

    def __init__(
        self,
        config: FastTextSearchClassifierConfig,
        embeddings_path: Optional[Pathlike] = None,
    ):
        """
        Initialize the classifier.

        Args:
            config: Configuration object with all parameters.
            embeddings_path: Optional path to FastText .bin file to load immediately.

        Raises:
            TypeError: If config is not FastTextSearchClassifierConfig.
        """
        if not isinstance(config, FastTextSearchClassifierConfig):
            raise TypeError(f"Expected FastTextSearchClassifierConfig, got {type(config)}")

        super().__init__(config)

        self.fasttext_model: Optional[_FastText] = None

        if embeddings_path is not None:
            self.load_embeddings(embeddings_path)

    def load_embeddings(self, model_path: Pathlike) -> Self:
        """
        Load pre-trained FastText embeddings.

        Download Polish FastText model from:
        https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz

        Args:
            model_path: Pathlike to FastText .bin file.

        Returns:
            self (for method chaining).
        """

        path: Path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"FastText model not found at: {str(model_path)}\n"
                f"Download Polish model:\n"
                f"wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz\n"
                f"gunzip cc.pl.300.bin.gz"
            )

        logger.info("Loading FastText embeddings from %s...", path)
        self.fasttext_model = fasttext.load_model(str(path))
        logger.info("Loaded embeddings (dim=%s)", self.fasttext_model.get_dimension())

        return self

    def _encode_text(self, text: str) -> npt.NDArray[np.floating]:
        """
        Convert text to vector representation using FastText embeddings.

        Uses mean pooling: averages word vectors to get sentence vector.

        Args:
            text: Input text

        Returns:
            Vector representation of shape (embedding_dim,)
        """
        self._check_embeddings_loaded()
        assert self.fasttext_model is not None

        words = text.lower().split()
        if len(words) == 0:
            return np.zeros(self.config.embedding_dim)

        vectors = [self.fasttext_model.get_word_vector(word) for word in words]
        if len(vectors) == 0:
            return np.zeros(self.config.embedding_dim)

        array: npt.NDArray[np.floating] = np.mean(vectors, axis=0)
        return array

    def _encode_batch(self, texts: Prompts) -> npt.NDArray[np.floating]:
        """
        Encode multiple texts to vector representations.

        Args:
            texts: List of input texts

        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        return np.array([self._encode_text(text) for text in texts])

    def train(self, dataset: Dataset, weights: Weights) -> Self:
        """
        Train the classifier on labeled data.

        Args:
            dataset: Dataset object containing prompts and labels
            weights: Dictionary mapping class indices to weights

        Returns:
            self (for method chaining)
        """
        self._check_embeddings_loaded()

        logger.info("Encoding %s prompts...", len(dataset.prompts))
        features = self._encode_batch(dataset.prompts)

        classifier_type = type(self.classifier).__name__
        logger.info("Training %s classifier...", classifier_type)
        fit_kwargs = self.prepare_sample_weights(weights, dataset.labels)
        self.classifier.fit(features, dataset.labels, **fit_kwargs)
        self._is_fitted = True
        logger.info("Model trained successfully")

        return self

    def predict(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.bool_]:
        """
        Predict whether prompts need web search.

        Args:
            prompts: Single prompt string or list of prompts.

        Returns:
            Boolean array (True = needs search, False = no search)
        """
        features = self._get_features(prompts)
        predictions: np.ndarray = self.classifier.predict(features)
        return predictions.astype(np.bool_)

    def predict_proba(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.floating]:
        """
        Predict probability of needing web search.

        Args:
            prompts: Single prompt string or list of prompts.

        Returns:
            Array of shape (n_samples, 2) with probabilities [no_search, needs_search]
        """
        features = self._get_features(prompts)
        proba: npt.NDArray[np.floating] = self.classifier.predict_proba(features)
        return proba

    def save(self, filepath: Pathlike) -> None:
        """
        Save the trained classifier (without embeddings).

        Note: This saves only the classifier, not the FastText embeddings.
        You'll need to load embeddings separately when loading the model.

        Args:
            filepath: Path to save the classifier (e.g., 'model.pkl')
        """
        save_dict = self._save_state(filepath)
        save_dict["classifier"] = self.classifier

        with open(filepath, "wb") as file:
            pickle.dump(save_dict, file)

        logger.info("Classifier saved to %s", filepath)
        logger.warning("FastText embeddings NOT included in save file. Load them separately when loading model.")

    @classmethod
    def load(cls, filepath: Pathlike, embeddings_path: Optional[Pathlike] = None, **kwargs: Any) -> Self:
        """
        Load a trained classifier and embeddings.

        Args:
            filepath: Path to the saved classifier
            embeddings_path: Path to FastText embeddings (.bin file)
            **kwargs: Additional arguments

        Returns:
            Loaded FastTextSearchClassifier instance
        """
        with open(filepath, "rb") as file:
            save_dict = pickle.load(file)

        model = cls(config=save_dict["config"])
        model._load_state(save_dict)
        model.classifier = save_dict["classifier"]

        if embeddings_path:
            model.load_embeddings(embeddings_path)
        else:
            logger.warning("Embeddings not loaded. Call load_embeddings() before using the model.")

        logger.info("Model loaded from %s", filepath)
        return model

    def _get_features(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.floating]:
        self._check_is_fitted()
        prompts = self._normalize_prompts(prompts)
        return self._encode_batch(prompts)

    @property
    def has_embeddings_loaded(self) -> bool:
        return self.fasttext_model is not None

    def _check_embeddings_loaded(self) -> None:
        """
        Check if embeddings have been loaded.

        Raises:
            RuntimeError: If embeddings haven't been loaded
        """
        if not self.has_embeddings_loaded:
            raise RuntimeError(
                "Embeddings not loaded. Call load_embeddings() first.\n"
                "Download Polish FastText: wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz"
            )

    def __repr__(self) -> str:
        classifier_type = type(self.classifier).__name__
        return (
            f"FastTextSearchClassifier("
            f"embedding_dim={self.config.embedding_dim}, "
            f"classifier={classifier_type}, "
            f"embeddings_loaded={self.has_embeddings_loaded}, "
            f"fitted={self._is_fitted}"
            f")"
        )
