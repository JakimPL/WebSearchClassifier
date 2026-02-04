from abc import ABC, abstractmethod
from typing import Generic

import numpy as np
import numpy.typing as npt

from websearchclassifier.config import ClassifierConfig
from websearchclassifier.utils import Kwargs, ProbabilisticClassifierT_co, Weights, logger


class ClassifierWrapper(Generic[ProbabilisticClassifierT_co], ABC):
    """
    A base class for scikit-learn classifiers to provide a consistent interface.

    Attributes:
        classifier (ProbabilisticClassifierT_co): The scikit-learn classifier instance.
    """

    classifier: ProbabilisticClassifierT_co

    def __init__(self, config: ClassifierConfig[ProbabilisticClassifierT_co]) -> None:
        self.config = config
        self.classifier = self.create()

    @abstractmethod
    def create(self) -> ProbabilisticClassifierT_co:
        """
        Create an instance of the classifier based on the configuration.

        Returns:
            ProbabilisticClassifierT_co: An instance of the classifier.
        """

    def apply_class_weights(
        self,
        weights: Weights,
        labels: npt.NDArray[np.bool_],
    ) -> Kwargs:
        """
        Apply class weights to the classifier.

        May be overridden by subclasses if specific handling is required.

        Args:
            weights (Weights): A dictionary mapping class indices to their weights.
            labels (npt.NDArray[np.bool_]): The labels corresponding to the training data.

        Returns:
            Kwargs: Additional keyword arguments for the `fit` method.
        """
        logger.info("Using class weights: no=%.2f, yes=%.2f", weights[0], weights[1])
        self.classifier.set_params(class_weight=weights)
        return {}

    @property
    @abstractmethod
    def feature_importances_(self) -> npt.NDArray[np.floating]:
        """
        Get feature importance scores, if supported.

        Raises:
            AttributeError: If the classifier does not support feature importance.
        """
