from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Self, Union

import numpy as np
import numpy.typing as npt

from websearchclassifier.classifier.type import load_classifier_wrapper
from websearchclassifier.config import ConfigT
from websearchclassifier.dataset import Dataset, DatasetLike, Labels, Prompts
from websearchclassifier.utils import Pathlike, ProbabilisticClassifier, Weights


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
        self.config: ConfigT = config
        self.wrapper = load_classifier_wrapper(config.classifier_config)
        self.classifier: ProbabilisticClassifier[Any] = self.wrapper.classifier

    def fit(
        self,
        dataset: Optional[DatasetLike] = None,
        prompts: Optional[Union[str, Prompts]] = None,
        labels: Optional[Union[bool, Labels]] = None,
    ) -> Self:
        """
        Train the classifier on labeled data.

        Accepts either prompts and labels directly, or a dataset object.

        Args:
            dataset: Dataset containing prompts and labels.
            prompts: List of prompt strings.
            labels: List of boolean labels (True = needs search, False = no search).

        Returns:
            self (for method chaining)
        """
        dataset = Dataset.create(dataset=dataset, prompts=prompts, labels=labels)
        weights = dataset.compute_class_weights()
        return self.train(dataset, weights)

    @abstractmethod
    def train(self, dataset: Dataset, weights: Weights) -> Self:
        """
        Train the classifier on the provided dataset.

        Args:
            dataset: Dataset containing prompts and labels

        Returns:
            self (for method chaining)
        """

    @abstractmethod
    def predict(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.bool_]:
        """
        Predict whether prompts need web search.

        Args:
            prompts: Single prompt string or list of prompts

        Returns:
            Boolean array (True = needs search, False = no search)
        """

    @abstractmethod
    def predict_proba(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.floating]:
        """
        Predict probability of needing web search.

        Args:
            prompts: Single prompt string or list of prompts

        Returns:
            Array of shape (n_samples, 2) with probabilities [no_search, needs_search]
        """

    def _save_state(self, filepath: Pathlike) -> Dict[str, Any]:
        """
        Get the base state dictionary for saving.

        Args:
            filepath: Path to save the model

        Returns:
            Dictionary with base state (config, _is_fitted)
        """
        self._check_is_fitted()
        return {
            "config": self.config,
            "_is_fitted": self._is_fitted,
        }

    def _load_state(self, save_dict: Dict[str, Any]) -> None:
        """
        Load the base state from a saved dictionary.

        Args:
            save_dict: Dictionary containing saved state
        """
        self._is_fitted = save_dict["_is_fitted"]

    @abstractmethod
    def save(self, filepath: Pathlike) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """

    @classmethod
    @abstractmethod
    def load(cls, filepath: Pathlike, **kwargs: Any) -> Self:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
            **kwargs: Additional arguments specific to the implementation

        Returns:
            Loaded classifier instance
        """

    @property
    def is_fitted(self) -> bool:
        """
        Check if the model has been fitted.

        Returns:
            True if the model has been fitted, False otherwise.
        """
        return self._is_fitted

    def _check_is_fitted(self) -> None:
        """
        Validate that the model has been fitted.

        Raises:
            RuntimeError: If model has not been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def _normalize_prompts(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.str_]:
        """
        Normalize input to always be a list of strings.

        Args:
            prompts: Single prompt or list of prompts

        Returns:
            List of prompt strings
        """
        return Dataset.normalize_prompts(prompts)

    def prepare_sample_weights(
        self,
        weights: Weights,
        labels: npt.NDArray[np.bool_],
    ) -> Dict[str, Any]:
        """
        Prepare sample weights for training.

        This method performs common checks and delegates to implementation-specific
        logic. Implementations configure the classifier and return kwargs for fit().

        Args:
            weights: Dictionary mapping class indices to weights
            labels: Boolean array of labels

        Returns:
            Dictionary of kwargs to pass to fit() method (empty dict if no weights)
        """
        if not self.config.use_class_weights:
            return {}

        if np.allclose(np.array(list(weights.values())), 1.0):
            return {}

        return self.wrapper.apply_class_weights(weights, labels)
