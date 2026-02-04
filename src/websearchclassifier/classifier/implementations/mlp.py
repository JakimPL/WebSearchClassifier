import numpy as np
import numpy.typing as npt
from sklearn.neural_network import MLPClassifier

from websearchclassifier.classifier.base import Classifier
from websearchclassifier.config import MLPConfig
from websearchclassifier.utils import Kwargs, Weights, logger


class MLPWrapper(Classifier[MLPClassifier]):
    """
    A wrapper for scikit-learn MLP classifier.

    Uses sklearn's MLPClassifier which provides a simple multi-layer perceptron
    implementation with backpropagation.

    Attributes:
        classifier (MLPClassifier): The scikit-learn MLP classifier instance.
    """

    config: MLPConfig
    classifier: MLPClassifier

    def create(self) -> MLPClassifier:
        """
        Create an instance of the MLP classifier based on the configuration.

        Returns:
            MLPClassifier: An instance of the MLP classifier.
        """
        return MLPClassifier(
            hidden_layer_sizes=tuple(self.config.hidden_layer_sizes),
            activation=self.config.activation,
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.max_iterations,
            batch_size=self.config.batch_size,
            random_state=self.config.random_state,
            early_stopping=self.config.early_stopping,
            validation_fraction=self.config.validation_fraction,
        )

    def apply_class_weights(
        self,
        weights: Weights,
        labels: npt.NDArray[np.bool_],
    ) -> Kwargs:
        sample_weights = np.array([weights[int(label)] for label in labels], dtype=np.float64)
        logger.info("Using sample weights: no=%.2f, yes=%.2f", weights[0], weights[1])
        return {"classifier__sample_weight": sample_weights}

    @property
    def feature_importances_(self) -> npt.NDArray[np.floating]:
        """
        Get feature importance scores, if supported.

        MLPClassifier does not provide feature importance scores.

        Raises:
            AttributeError: Always, since MLP does not support feature importance.
        """
        raise AttributeError("Feature importances are not available for MLPClassifier.")
