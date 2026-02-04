from __future__ import annotations

from typing import Any, Generic, Optional, Self, Union

import numpy as np
import numpy.typing as npt
from sklearn.pipeline import Pipeline

from websearchclassifier.baseline import Baseline, load_baseline
from websearchclassifier.classifier import Classifier, load_classifier_wrapper
from websearchclassifier.config import BaselineConfig, ClassifierConfig, WebSearchClassifierConfig
from websearchclassifier.dataset import Dataset, DatasetLike, Label, Labels, Prompt, Prompts
from websearchclassifier.utils import (
    Kwargs,
    ModelT,
    Pathlike,
    ProbabilisticClassifier,
    ProbabilisticClassifierT_co,
    Weights,
    load_pickle,
    logger,
    save_pickle,
)


class WebSearchClassifier(Generic[ModelT, ProbabilisticClassifierT_co]):
    """
    Binary classifier determining if a prompt requires web search.

    Combines a baseline for embedding generation with a probabilistic classifier.

    General scheme of the pipeline:
    ```
              baseline         classifier
        prompt    →   embedding     →    prediction
    ```
    """

    def __init__(
        self,
        config: WebSearchClassifierConfig,
    ) -> None:
        """
        Initialize the classifier.

        Args:
            baseline_config: Configuration object for the baseline.
            classifier_config: Configuration object for the classifier.

        Raises:
            TypeError: If config is not WebSearchClassifierConfig.
        """
        if not isinstance(config, WebSearchClassifierConfig):
            raise TypeError(f"Expected WebSearchClassifierConfig, got {type(config)}")

        self._is_fitted: bool = False
        self.config: WebSearchClassifierConfig = config
        self.baseline: Baseline[ModelT] = load_baseline(config.baseline)
        self.wrapper: Classifier[ProbabilisticClassifierT_co] = load_classifier_wrapper(self.classifier_config)
        self.pipeline: Pipeline = Pipeline([("classifier", self.classifier)])

    def fit(
        self,
        dataset: Optional[DatasetLike] = None,
        prompts: Optional[Union[Prompt, Prompts]] = None,
        labels: Optional[Union[Label, Labels]] = None,
    ) -> Self:
        """
        Train the classifier on labeled data.

        Accepts either a Dataset object or separate prompts and labels.

        Args:
            dataset: Dataset containing prompts and labels.
            prompts: A single prompt or a list of prompt strings.
            labels: A single label or a list of labels.

        Returns:
            self (for method chaining).
        """
        dataset = Dataset.create(dataset=dataset, prompts=prompts, labels=labels)
        weights = dataset.compute_class_weights()
        return self.train(dataset, weights)

    def train(self, dataset: Dataset, weights: Weights) -> Self:
        """
        Train the classifier on the provided dataset.

        Args:
            dataset: Dataset containing prompts and labels.
            weights: Dictionary mapping class indices to weights.

        Returns:
            self (for method chaining).
        """
        if self.baseline.is_trainable:
            logger.info("Training baseline on %d samples...", len(dataset.prompts))
            self.baseline.fit(dataset.prompts)

        logger.info("Encoding %d prompts...", len(dataset.prompts))
        features = self.baseline.transform(dataset.prompts)

        classifier_name = type(self.classifier).__name__
        logger.info("Training %s classifier...", classifier_name)
        fit_kwargs = self._prepare_sample_weights(weights, dataset.labels)
        self.pipeline.fit(features, dataset.labels, **fit_kwargs)
        self._is_fitted = True
        logger.info("Classifier trained successfully")
        return self

    def embeddings(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.floating]:
        """Get vector embeddings for the given prompts.

        Args:
            prompts: Single prompt string or list of prompts.

        Returns:
            Array with prompt vector representations.
        """
        self._check_is_fitted()
        normalized = self.normalize_prompts(prompts)
        return self.baseline.transform(normalized)

    def predict(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.bool_]:
        """Predict whether prompts need web search.

        Args:
            prompts: Single prompt string or list of prompts.

        Returns:
            Boolean array (True = needs search, False = no search).
        """
        features = self.embeddings(prompts)
        predictions: np.ndarray = self.classifier.predict(features)
        return predictions.astype(np.bool_)

    def predict_proba(self, prompts: Union[str, Prompts]) -> npt.NDArray[np.floating]:
        """Predict probability of needing web search.

        Args:
            prompts: Single prompt string or list of prompts.

        Returns:
            Array of shape (n_samples, 2) with probabilities [no_search, needs_search].
        """
        features = self.embeddings(prompts)
        proba: npt.NDArray[np.floating] = self.classifier.predict_proba(features)
        return proba

    def save(self, filepath: Pathlike) -> None:
        """Save the trained classifier to disk.

        Args:
            filepath: Path to save the classifier.
        """
        self._check_is_fitted()
        data = {
            "config": self.config,
            "wrapper": self.wrapper,
            "_is_fitted": self._is_fitted,
        }

        if self.is_baseline_trainable:
            data["baseline"] = self.baseline

        save_pickle(data, filepath)
        logger.info("Classifier saved to: '%s'", filepath)

    @classmethod
    def load(cls, filepath: Pathlike, **kwargs: Any) -> Self:
        """Load a trained classifier from disk.

        Args:
            filepath: Path to the saved classifier.

        Returns:
            Loaded classifier instance.
        """
        data = load_pickle(filepath)
        self = cls(config=data["config"])
        self.config = data["config"]
        self.wrapper = data["wrapper"]
        self._is_fitted = data["_is_fitted"]

        if self.is_baseline_trainable:
            self.baseline = data["baseline"]

        logger.info("Classifier loaded from: '%s'", filepath)
        return self

    @property
    def model(self) -> Any:
        """
        Get the underlying model instance from the wrapper.

        Returns:
            The model wrapped by the Baseline.
        """
        return self.baseline.model

    @property
    def classifier(self) -> ProbabilisticClassifier[Any]:
        """
        Get the underlying classifier instance.

        Returns:
            The classifier wrapped by the ClassifierWrapper.
        """
        return self.wrapper.classifier

    @property
    def baseline_config(self) -> BaselineConfig:
        """
        Get the baseline configuration.

        Returns:
            The baseline configuration object.
        """
        return self.config.baseline

    @property
    def classifier_config(self) -> ClassifierConfig:
        """
        Get the classifier configuration.

        Returns:
            The classifier configuration object.
        """
        return self.config.classifier

    @property
    def is_baseline_trainable(self) -> bool:
        """
        Check if the baseline is trainable.

        Returns:
            True if the baseline is trainable, False otherwise.
        """
        return self.baseline.is_trainable

    @property
    def is_fitted(self) -> bool:
        """
        Return whether the classifier has been fitted.

        Returns:
            True if fitted, False otherwise.
        """
        return self._is_fitted

    def _check_is_fitted(self) -> None:
        """
        Validate that the model has been fitted.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.baseline.is_trainable and not self.baseline.is_fitted:
            raise RuntimeError("Baseline model has not been fitted yet. Call fit() first.")

        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def normalize_prompts(self, prompts: Union[Prompt, Prompts]) -> npt.NDArray[np.str_]:
        """
        Normalize input to be an array of strings.

        Args:
            prompts: Single prompt string or list of prompt strings.

        Returns:
            Array of prompt strings.
        """
        return Dataset.normalize_prompts(prompts)

    def _prepare_sample_weights(self, weights: Weights, labels: npt.NDArray[np.bool_]) -> Kwargs:
        """
        Prepare sample weights for training, and return additional `fit` kwargs
        for the classifier, if needed.

        If class weights are not used or are all 1.0, returns an empty dict.

        Args:
            weights: Dictionary mapping class indices to weights.
            labels: Array of labels corresponding to the training data.

        Returns:
            Additional keyword arguments for the classifier's `fit` method.
        """
        if not self.config.use_class_weights:
            return {}

        if np.allclose(np.array(list(weights.values())), 1.0):
            return {}

        return self.wrapper.apply_class_weights(weights, labels)
