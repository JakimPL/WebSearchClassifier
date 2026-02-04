from __future__ import annotations

from typing import Any, List, Self, Tuple, Union

import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from websearchclassifier.config import TFIDFSearchClassifierConfig
from websearchclassifier.dataset import Dataset, Prompts
from websearchclassifier.model.base import SearchClassifier
from websearchclassifier.utils import Pathlike, Weights, logger


class TFIDFSearchClassifier(SearchClassifier[TFIDFSearchClassifierConfig]):
    """
    Binary classifier for determining if a prompt requires web search.

    Uses TF-IDF vectorization with Logistic Regression for fast, accurate
    classification without requiring external LLM.

    Example:
        >>> classifier = TFIDFSearchClassifier()
        >>> # Training data: (prompt, needs_search)
        >>> train_data = [
        ...     ("jaka jest dzisiaj pogoda w Warszawie", True),
        ...     ("oblicz pierwiastek z 144", False),
        ...     ("kto wygrał ostatni mundial", True),
        ...     ("napisz wiersz o kocie", False),
        ... ]
        >>> prompts, labels = zip(*train_data)
        >>> classifier.fit(prompts, labels)
        >>>
        >>> # Prediction
        >>> classifier.predict(["jaka jest aktualna cena bitcoina"])
        array([True])
        >>> classifier.predict_proba(["co to jest rekurencja"])
        array([[0.35, 0.65]])  # [no_search, needs_search]
    """

    def __init__(
        self,
        config: TFIDFSearchClassifierConfig,
    ):
        """
        Initialize the classifier.

        Args:
            config: Configuration object with all parameters.
        """
        if not isinstance(config, TFIDFSearchClassifierConfig):
            raise TypeError(f"Expected TFIDFSearchClassifierConfig, got {type(config)}")

        self.vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            min_df=config.min_document_frequency,
            max_df=config.max_document_frequency,
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b\w+\b",
        )

        super().__init__(config)

    def train(self, dataset: Dataset, weights: Weights) -> Self:
        """
        Train the classifier on labeled data.

        Args:
            dataset: Dataset containing prompts and labels.
            weights: Dictionary mapping class indices to weights.

        Returns:
            self (for method chaining).
        """
        return self._train(dataset.prompts, dataset.labels, weights)

    def predict(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.bool_]:
        """
        Predict whether prompts need web search.

        Args:
            prompts: Single prompt string or list of prompts.

        Returns:
            Boolean array (True = needs search, False = no search).
        """
        features: npt.NDArray[np.str_] = self._get_features(prompts)
        result = self.pipeline.predict(features)
        assert isinstance(result, np.ndarray)
        predictions: npt.NDArray[np.bool_] = result.astype(np.bool_)
        return predictions

    def predict_proba(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.floating]:
        """
        Predict probability of needing web search.

        Args:
            prompts: Single prompt string or list of prompts.

        Returns:
            Array of shape (n_samples, 2) with probabilities [no_search, needs_search].
        """
        features: npt.NDArray[np.str_] = self._get_features(prompts)
        proba: npt.NDArray[np.floating] = self.pipeline.predict_proba(features)
        return proba

    def get_feature_importance(self, top_n: int = 20) -> Tuple[List[str], List[str]]:
        """
        Get most important features (words/n-grams) for each class.

        Args:
            top_n: Number of top features to return.

        Returns:
            Tuple of (top_no_search_features, top_needs_search_features).

        Raises:
            RuntimeError: If the classifier is not fitted.
            ValueError: If the classifier does not have coefficients.
            ValueError: If feature importance is not supported for this classifier type.
        """
        self.check_is_fitted()

        feature_names: npt.NDArray[np.str_] = self.vectorizer.get_feature_names_out()
        coefficients = self.wrapper.feature_importances_
        sorted_indices = np.argsort(coefficients)

        top_no_search: List[str] = [str(feature_names[i]) for i in sorted_indices[:top_n]]
        top_needs_search: List[str] = [str(feature_names[i]) for i in sorted_indices[-top_n:][::-1]]

        return top_no_search, top_needs_search

    def save(self, filepath: Pathlike) -> None:
        """
        Save the trained TF-IDF classifier to disk.

        Args:
            filepath: Path to the saved classifier.
        """
        self._save_state(filepath)
        logger.info("TF-IDF classifier saved to: '%s'", filepath)

    @classmethod
    def load(cls, filepath: Pathlike, **kwargs: Any) -> Self:
        """
        Load a trained TF-IDF classifier from disk.

        Args:
            filepath: Path to the saved classifier.

        Returns:
            Loaded TFIDFSearchClassifier instance.
        """
        model = cls._load_state(filepath)
        logger.info("TF-IDF classifier loaded from: '%s'", filepath)
        return model

    def _create_pipeline(self) -> Pipeline:
        return Pipeline([("tfidf", self.vectorizer), ("classifier", self.classifier)])

    def _get_features(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.str_]:
        self.check_is_fitted()
        return self.normalize_prompts(prompts)

    def __repr__(self) -> str:
        classifier_type = type(self.classifier).__name__
        return (
            f"TFIDFSearchClassifier("
            f"classifier={classifier_type}, "
            f"max_features={self.config.max_features}, "
            f"ngram_range={self.config.ngram_range}, "
            f"fitted={self._is_fitted}"
            f")"
        )
