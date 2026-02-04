from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Self, Union

import numpy as np
import numpy.typing as npt
from sklearn.pipeline import Pipeline

from websearchclassifier.classifier import ClassifierWrapper, load_classifier_wrapper
from websearchclassifier.config import ConfigT
from websearchclassifier.dataset import Dataset, DatasetLike, Labels, Prompts
from websearchclassifier.utils import Pathlike, ProbabilisticClassifier, Weights, load_pickle, logger, save_pickle
from websearchclassifier.utils.types import Kwargs


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

        Args:
            config: Configuration object with all parameters.
        """
        self._is_fitted: bool = False
        self.config: ConfigT = config
        self.wrapper: ClassifierWrapper[Any] = load_classifier_wrapper(config.classifier_config)
        self.pipeline: Pipeline = self._create_pipeline()

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
            self (for method chaining).
        """
        dataset = Dataset.create(dataset=dataset, prompts=prompts, labels=labels)
        weights = dataset.compute_class_weights()
        return self.train(dataset, weights)

    @abstractmethod
    def train(self, dataset: Dataset, weights: Weights) -> Self:
        """
        Train the classifier on the provided dataset.

        Args:
            dataset: Dataset containing prompts and labels.

        Returns:
            self (for method chaining).
        """

    def _train(
        self,
        features: npt.NDArray[Union[np.floating, np.str_]],
        labels: npt.NDArray[np.bool_],
        weights: Weights,
    ) -> Self:
        """
        A common training routine used by implementations.

        Args:
            features: Feature matrix for training.
            labels: Labels corresponding to the features.
            weights: Dictionary mapping class indices to weights.

        Returns:
            self (for method chaining).
        """
        classifier_name = type(self.classifier).__name__
        logger.info("Training %s classifier on %s samples...", classifier_name, len(labels))
        fit_kwargs = self.prepare_sample_weights(weights, labels)
        self.pipeline.fit(features, labels, **fit_kwargs)
        self._is_fitted = True
        logger.info("Classifier trained successfully")
        return self

    @abstractmethod
    def predict(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.bool_]:
        """
        Predict whether prompts need web search.

        Args:
            prompts: Single prompt string or list of prompts.

        Returns:
            Boolean array (True = needs search, False = no search).
        """

    @abstractmethod
    def predict_proba(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.floating]:
        """
        Predict probability of needing web search.

        Args:
            prompts: Single prompt string or list of prompts.

        Returns:
            Array of shape (n_samples, 2) with probabilities [no_search, needs_search].
        """

    def _create_pipeline(self) -> Pipeline:
        """
        Create the training pipeline specific to the classifier implementation.

        Returns a pipeline consisting of a single classifier by default.

        Returns:
            Pipeline instance.
        """
        return Pipeline([("classifier", self.classifier)])

    def _save_state(self, filepath: Pathlike) -> None:
        """
        Save the base state to a dictionary.

        Args:
            filepath: Path to save the model.
        """
        self.check_is_fitted()
        data = {
            "pipeline": self.pipeline,
            "wrapper": self.wrapper,
            "config": self.config,
            "_is_fitted": self._is_fitted,
        }

        save_pickle(data, filepath)

    @classmethod
    def _load_state(cls, filepath: Pathlike) -> Self:
        """
        Load the base state from a saved dictionary.

        Args:
            filepath: Path to the saved classifier.

        Returns:
            Loaded classifier instance.
        """
        data = load_pickle(filepath)
        self = cls(config=data["config"])
        self._is_fitted = data["_is_fitted"]
        self.pipeline = data["pipeline"]
        self.wrapper = data["wrapper"]
        return self

    @abstractmethod
    def save(self, filepath: Pathlike) -> None:
        """
        Save the trained classifier to disk.

        Args:
            filepath: Path to save the classifier
        """

    @classmethod
    @abstractmethod
    def load(cls, filepath: Pathlike, **kwargs: Any) -> Self:
        """
        Load a trained classifier from disk.

        Args:
            filepath: Path to the saved classifier.
            **kwargs: Additional arguments specific to the implementation.

        Returns:
            Loaded classifier instance.
        """

    @property
    def classifier(self) -> ProbabilisticClassifier[Any]:
        """
        Get the underlying classifier instance.

        Returns:
            Classifier instance.
        """
        classifier: ProbabilisticClassifier[Any] = self.wrapper.classifier
        return classifier

    @property
    def is_fitted(self) -> bool:
        """
        Check if the model has been fitted.

        Returns:
            True if the model has been fitted, False otherwise.
        """
        return self._is_fitted

    def check_is_fitted(self) -> None:
        """
        Validate that the model has been fitted.

        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def normalize_prompts(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.str_]:
        """
        Normalize input to always be a list of strings.

        Args:
            prompts: Single prompt or list of prompts.

        Returns:
            List of prompt strings.
        """
        return Dataset.normalize_prompts(prompts)

    def prepare_sample_weights(self, weights: Weights, labels: npt.NDArray[np.bool_]) -> Kwargs:
        """
        Prepare sample weights for training.

        This method performs common checks and delegates to implementation-specific
        logic. Implementations configure the classifier and return kwargs for fit().

        Args:
            weights: Dictionary mapping class indices to weights.
            labels: Array of boolean labels corresponding to the samples.

        Returns:
            Kwargs: Additional keyword arguments for the `fit` method.
        """
        if not self.config.use_class_weights:
            return {}

        if np.allclose(np.array(list(weights.values())), 1.0):
            return {}

        return self.wrapper.apply_class_weights(weights, labels)
