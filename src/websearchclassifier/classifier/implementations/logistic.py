import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression

from websearchclassifier.classifier.wrapper import ClassifierWrapper
from websearchclassifier.config import LogisticRegressionConfig


class LogisticRegressionWrapper(ClassifierWrapper[LogisticRegression]):
    """
    A wrapper for scikit-learn Logistic Regression classifier.

    Attributes:
        classifier (LogisticRegression): The scikit-learn Logistic Regression classifier instance.
    """

    config: LogisticRegressionConfig
    classifier: LogisticRegression

    def create(self) -> LogisticRegression:
        """
        Create an instance of the Logistic Regression classifier based on the configuration.

        Returns:
            LogisticRegression: An instance of the Logistic Regression classifier.
        """
        return LogisticRegression(
            C=self.config.regularization_strength,
            random_state=self.config.random_state,
            max_iter=self.config.max_iterations,
            solver=self.config.solver,
        )

    @property
    def feature_importances_(self) -> npt.NDArray[np.floating]:
        """
        Get logistic regression coefficients as feature importance scores.

        Returns:
            npt.NDArray[np.floating]: Absolute values of logistic regression coefficients.
        """
        coefficients: npt.NDArray[np.floating] = np.abs(self.classifier.coef_[0])
        return coefficients
