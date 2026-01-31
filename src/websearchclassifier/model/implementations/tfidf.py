"""
TF-IDF based binary classifier for web search decision.
Fast, lightweight classifier that learns automatically from training data.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Self, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from websearchclassifier.config import TfidfSearchClassifierConfig
from websearchclassifier.model.base import SearchClassifier


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

    def fit(self, prompts: List[str], labels: List[bool]) -> TfidfSearchClassifier:
        """
        Train the classifier on labeled data.

        Args:
            prompts: List of prompt strings
            labels: List of boolean labels (True = needs search, False = no search)

        Returns:
            self (for method chaining)
        """
        self._validate_training_data(prompts, labels)

        labels_array = np.array(labels, dtype=int)
        self.pipeline.fit(prompts, labels_array)
        self._is_fitted = True

        return self

    def predict(self, prompts: Union[str, List[str]]) -> np.ndarray:
        """
        Predict whether prompts need web search.

        Args:
            prompts: Single prompt string or list of prompts

        Returns:
            Boolean array (True = needs search, False = no search)
        """
        self._check_is_fitted()
        prompts = self._normalize_input(prompts)

        predictions: np.ndarray = self.pipeline.predict(prompts)
        return predictions.astype(bool)

    def predict_proba(self, prompts: Union[str, List[str]]) -> np.ndarray:
        """
        Predict probability of needing web search.

        Args:
            prompts: Single prompt string or list of prompts

        Returns:
            Array of shape (n_samples, 2) with probabilities [no_search, needs_search]
        """
        self._check_is_fitted()
        unified_prompts: List[str] = self._normalize_input(prompts)
        proba: np.ndarray = self.pipeline.predict_proba(unified_prompts)
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

        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.classifier.coef_[0]
        sorted_indices = np.argsort(coefficients)

        top_no_search = [feature_names[i] for i in sorted_indices[:top_n]]
        top_needs_search = [feature_names[i] for i in sorted_indices[-top_n:][::-1]]

        return top_no_search, top_needs_search

    def save(self, filepath: Path) -> None:
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

    @classmethod
    def load(cls, filepath: Path, **kwargs: Any) -> Self:
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

        return model

    def __repr__(self) -> str:
        return (
            f"TfidfSearchClassifier("
            f"max_features={self.config.max_features}, "
            f"ngram_range={self.config.ngram_range}, "
            f"regularization={self.config.regularization_strength}, "
            f"fitted={self._is_fitted}"
            f")"
        )
