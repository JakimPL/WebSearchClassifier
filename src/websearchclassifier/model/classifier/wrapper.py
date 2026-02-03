from abc import ABC, abstractmethod
from typing import Any, Dict, Generic

import numpy as np
import numpy.typing as npt

from websearchclassifier.config import ClassifierConfig
from websearchclassifier.utils import ProbabilisticClassifierT_co, Weights


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

    @abstractmethod
    def apply_class_weights(
        self,
        weights: Weights,
        labels: npt.NDArray[np.bool_],
    ) -> Dict[str, Any]:
        """
        Apply class weights to the classifier.

        Args:
            weights (Weights): A dictionary mapping class labels to their weights.
            labels (npt.NDArray[np.bool_]): An array of labels corresponding to the training data.

        Returns:
            Dict[str, Any]: Additional keyword arguments to be passed to the `fit` method of the classifier.
        """
