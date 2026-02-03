from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression

from websearchclassifier.classifier.wrapper import ClassifierWrapper
from websearchclassifier.config import LogisticRegressionConfig
from websearchclassifier.utils import Weights, logger


class LogisticRegressionWrapper(ClassifierWrapper[LogisticRegression]):
    """
    A wrapper for scikit-learn Logistic Regression classifier.

    Attributes:
        classifier (LogisticRegression): The scikit-learn Logistic Regression classifier instance.
    """

    config: LogisticRegressionConfig
    classifier: LogisticRegression

    def create(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.config.regularization_strength,
            random_state=self.config.random_state,
            max_iter=self.config.max_iterations,
            solver=self.config.solver,
        )

    def apply_class_weights(
        self,
        weights: Weights,
        labels: npt.NDArray[np.bool_],
    ) -> Dict[str, Any]:
        logger.info("Using class weights: %s", weights)
        self.classifier.set_params(class_weight=weights)
        return {}
