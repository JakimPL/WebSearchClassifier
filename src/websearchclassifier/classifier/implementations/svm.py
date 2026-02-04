import numpy as np
import numpy.typing as npt
from sklearn.svm import SVC

from websearchclassifier.classifier.base import Classifier
from websearchclassifier.config import SVMConfig


class SVMWrapper(Classifier[SVC]):
    """
    A wrapper for scikit-learn SVM classifier.

    Attributes:
        classifier (SVC): The scikit-learn SVM classifier instance.
    """

    config: SVMConfig
    classifier: SVC

    def create(self) -> SVC:
        """
        Create an instance of the SVM classifier based on the configuration.

        Returns:
            SVC: An instance of the SVM classifier.
        """
        return SVC(
            C=self.config.regularization_strength,
            random_state=self.config.random_state,
            kernel=self.config.kernel,
            probability=self.config.probability,
        )

    @property
    def feature_importances_(self) -> npt.NDArray[np.floating]:
        """
        Get feature importance scores for linear kernel SVM. Other kernels do not support feature importance.

        Raises:
            AttributeError: If the kernel is not linear.
        """
        if self.config.kernel == "linear":
            feature_importances: npt.NDArray[np.floating] = np.abs(self.classifier.coef_[0])
            return feature_importances

        raise AttributeError(
            f"Feature importances are not available for kernel='{self.config.kernel}'. "
            "Only 'linear' kernel supports feature importances."
        )
