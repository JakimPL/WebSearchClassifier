from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from sklearn.svm import SVC

from websearchclassifier.config import SVMConfig
from websearchclassifier.model.classifier.wrapper import ClassifierWrapper
from websearchclassifier.utils import Weights, logger


class SVMWrapper(ClassifierWrapper[SVC]):
    """
    A wrapper for scikit-learn SVM classifier.

    Attributes:
        classifier (SVC): The scikit-learn SVM classifier instance.
    """

    config: SVMConfig
    classifier: SVC

    def create(self) -> SVC:
        return SVC(
            C=self.config.regularization_strength,
            random_state=self.config.random_state,
            kernel=self.config.kernel,
            probability=self.config.probability,
        )

    def apply_class_weights(
        self,
        weights: Weights,
        labels: npt.NDArray[np.bool_],
    ) -> Dict[str, Any]:
        fit_kwargs: Dict[str, Any] = {}
        sample_weights = np.array([weights[int(label)] for label in labels], dtype=np.float64)
        logger.info("Using sample weights (mean: %.3f)", sample_weights.mean())
        fit_kwargs["sample_weight"] = sample_weights
        return fit_kwargs
