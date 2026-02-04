from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np
import numpy.typing as npt

from websearchclassifier.config import BaselineConfig
from websearchclassifier.utils import ModelT


class Baseline(ABC, Generic[ModelT]):
    """
    Abstract base class for embedding generation (baseline models).

    A baseline transforms text prompts into numeric representations
    that can be fed into a classifier.
    """

    model: ModelT

    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if the baseline has been fitted/loaded."""
        return self._is_fitted

    @abstractmethod
    def fit(self, prompts: npt.NDArray[np.str_]) -> None:
        """Train the baseline on prompts (if trainable).

        Args:
            prompts: Array of prompt strings
        """

    @abstractmethod
    def transform(self, prompts: npt.NDArray[np.str_]) -> npt.NDArray[np.floating]:
        """Transform prompts to embeddings.

        Args:
            prompts: Array of prompt strings

        Returns:
            Array of embeddings, shape (n_samples, embedding_dim)
        """

    @property
    @abstractmethod
    def is_trainable(self) -> bool:
        """
        Whether this baseline requires training.

        Returns:
            True if the baseline is trainable, False otherwise.
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """
        Dimension of the embeddings produced.

        Returns:
            The embedding dimension as an integer.
        """

    @property
    def class_name(self) -> str:
        """
        Get the class name of the baseline.

        Returns:
            The class name as a string.
        """
        return self.__class__.__name__
