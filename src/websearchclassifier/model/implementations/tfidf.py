"""
TF-IDF based binary classifier for web search decision.
Fast, lightweight classifier that learns automatically from training data.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Self, Tuple, Union

import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from websearchclassifier.config import TfidfSearchClassifierConfig
from websearchclassifier.dataset import Dataset, Prompts
from websearchclassifier.model.base import SearchClassifier
from websearchclassifier.utils import Pathlike, logger


class TfidfSearchClassifier(SearchClassifier[TfidfSearchClassifierConfig]):
    """
    Binary classifier for determining if a prompt requires web search.

    Uses TF-IDF vectorization with Logistic Regression for fast, accurate
    classification without requiring external LLM.

    Example:
        >>> classifier = TfidfSearchClassifier()
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
        config: TfidfSearchClassifierConfig,
    ):
        """
        Initialize the classifier.

        Args:
            config: Configuration object with all parameters
        """
        super().__init__(config)

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

        self.classifier = LogisticRegression(
            C=config.regularization_strength,
            random_state=config.random_state,
            max_iter=1000,
            solver="liblinear",
        )

        self.pipeline = Pipeline([("tfidf", self.vectorizer), ("classifier", self.classifier)])
        self._is_fitted = False

    def _apply_class_weights(
        self,
        weights: Dict[int, float],
        labels: npt.NDArray[np.bool_],
    ) -> Dict[str, Any]:
        """
        Apply class weights for LogisticRegression.

        Args:
            weights: Dictionary mapping class indices to weights
            labels: Boolean array of labels (unused for class_weight approach)

        Returns:
            Empty dict (weights applied via set_params)
        """
        logger.info("Using class weights: %s", weights)
        self.classifier.set_params(class_weight=weights)
        return {}

    def train(self, dataset: Dataset, weights: Dict[int, float]) -> Self:
        """
        Train the classifier on labeled data.

        Args:
            dataset: Dataset containing prompts and labels
            weights: Dictionary mapping class indices to weights

        Returns:
            self (for method chaining)
        """
        logger.info("Training TF-IDF classifier on %s samples...", len(dataset.prompts))
        fit_kwargs = self.prepare_sample_weights(weights, dataset.labels)
        self.pipeline.fit(dataset.prompts, dataset.labels, **fit_kwargs)
        self._is_fitted = True
        logger.info("Model trained successfully")
        return self

    def predict(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.bool_]:
        """
        Predict whether prompts need web search.

        Args:
            prompts: Single prompt string or list of prompts

        Returns:
            Boolean array (True = needs search, False = no search)
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
            prompts: Single prompt string or list of prompts

        Returns:
            Array of shape (n_samples, 2) with probabilities [no_search, needs_search]
        """
        features: npt.NDArray[np.str_] = self._get_features(prompts)
        proba: npt.NDArray[np.floating] = self.pipeline.predict_proba(features)
        return proba

    def get_feature_importance(self, top_n: int = 20) -> Tuple[List[str], List[str]]:
        """
        Get most important features (words/n-grams) for each class.

        Args:
            top_n: Number of top features to return

        Returns:
            Tuple of (top_no_search_features, top_needs_search_features)
        """
        self._check_is_fitted()

        feature_names: npt.NDArray[np.str_] = self.vectorizer.get_feature_names_out()
        coefficients = self.classifier.coef_[0]
        sorted_indices = np.argsort(coefficients)

        top_no_search: List[str] = [str(feature_names[i]) for i in sorted_indices[:top_n]]
        top_needs_search: List[str] = [str(feature_names[i]) for i in sorted_indices[-top_n:][::-1]]

        return top_no_search, top_needs_search

    def save(self, filepath: Pathlike) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model (e.g., 'model.pkl')
        """
        self._check_is_fitted()

        save_dict = {
            "config": self.config,
            "pipeline": self.pipeline,
            "_is_fitted": self._is_fitted,
        }

        with open(filepath, "wb") as file:
            pickle.dump(save_dict, file)

        logger.info("Classifier saved to %s", filepath)

    @classmethod
    def load(cls, filepath: Pathlike, **kwargs: Any) -> Self:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
            **kwargs: Unused, kept for interface compatibility

        Returns:
            Loaded TfidfSearchClassifier instance
        """
        with open(filepath, "rb") as file:
            save_dict = pickle.load(file)

        if isinstance(save_dict, cls):
            return save_dict

        model = cls(config=save_dict["config"])
        model.pipeline = save_dict["pipeline"]
        model._is_fitted = save_dict["_is_fitted"]

        logger.info("Model loaded from %s", filepath)
        return model

    def _get_features(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.str_]:
        self._check_is_fitted()
        return self._normalize_prompts(prompts)

    def __repr__(self) -> str:
        return (
            f"TfidfSearchClassifier("
            f"max_features={self.config.max_features}, "
            f"ngram_range={self.config.ngram_range}, "
            f"regularization={self.config.regularization_strength}, "
            f"fitted={self._is_fitted}"
            f")"
        )
