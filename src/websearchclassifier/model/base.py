from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, List, Self, TypeVar, Union

import numpy as np

from websearchclassifier.config import SearchClassifierConfig

ConfigT = TypeVar("ConfigT", bound=SearchClassifierConfig)


class SearchClassifier(ABC, Generic[ConfigT]):
    """
    Abstract base class for binary classifiers determining if a prompt requires web search.

    All classifier implementations must inherit from this class and implement
    the required methods to ensure consistent interface across different models.

    This enables easy model comparison, evaluation, and switching between
    different classification approaches (TF-IDF, embeddings, tree-based, etc.).
    """

    def __init__(self, config: ConfigT) -> None:
        """
        Initialize the base classifier.
        """
        self._is_fitted: bool = False
        self.config = config

    @abstractmethod
    def fit(self, prompts: List[str], labels: List[bool]) -> Self:
        """
        Train the classifier on labeled data.

        Args:
            prompts: List of prompt strings
            labels: List of boolean labels (True = needs search, False = no search)

        Returns:
            self (for method chaining)
        """

    @abstractmethod
    def predict(self, prompts: Union[str, List[str]]) -> np.ndarray:
        """
        Predict whether prompts need web search.

        Args:
            prompts: Single prompt string or list of prompts

        Returns:
            Boolean array (True = needs search, False = no search)
        """

    @abstractmethod
    def predict_proba(self, prompts: Union[str, List[str]]) -> np.ndarray:
        """
        Predict probability of needing web search.

        Args:
            prompts: Single prompt string or list of prompts

        Returns:
            Array of shape (n_samples, 2) with probabilities [no_search, needs_search]
        """

    @abstractmethod
    def save(self, filepath: Path) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """

    @classmethod
    @abstractmethod
    def load(cls, filepath: Path, **kwargs: Any) -> Self:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
            **kwargs: Additional arguments specific to the implementation

        Returns:
            Loaded classifier instance
        """

    def _check_is_fitted(self) -> None:
        """
        Check if the model has been fitted.

        Raises:
            RuntimeError: If model has not been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def _normalize_input(self, prompts: Union[str, List[str]]) -> List[str]:
        """
        Normalize input to always be a list of strings.

        Args:
            prompts: Single prompt or list of prompts

        Returns:
            List of prompt strings
        """
        if isinstance(prompts, str):
            return [prompts]

        return prompts

    def _validate_training_data(self, prompts: List[str], labels: List[bool]) -> None:
        """
        Validate training data before fitting.

        Args:
            prompts: List of prompt strings
            labels: List of boolean labels

        Raises:
            ValueError: If data is invalid
        """
        if len(prompts) != len(labels):
            raise ValueError(f"prompts and labels must have same length: {len(prompts)} != {len(labels)}")

        if len(prompts) == 0:
            raise ValueError("Cannot fit on empty dataset")
