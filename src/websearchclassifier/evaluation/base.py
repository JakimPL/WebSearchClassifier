from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from websearchclassifier.config import DatasetConfig, EvaluatorConfig
from websearchclassifier.dataset import Dataset
from websearchclassifier.evaluation.types import Metric
from websearchclassifier.model import SearchClassifier

EvaluatorConfigT = TypeVar("EvaluatorConfigT", bound=EvaluatorConfig)


class Evaluator(ABC, Generic[EvaluatorConfigT]):
    def __init__(self, config: EvaluatorConfigT) -> None:
        self.config = config

    @abstractmethod
    def __call__(
        self,
        model: SearchClassifier[Any],
        dataset: Dataset,
        metric: Metric,
        **init_kwargs: Any,
    ) -> float:
        """
        Evaluate the given dataset using the specified metric.

        Args:
            model (SearchClassifier): The model to evaluate.
            dataset (Dataset): The dataset to evaluate.
            metric (Metric): The metric to use for evaluation.
            **init_kwargs: Additional keyword arguments for model initialization.

        Returns:
            float: The evaluation score.
        """

    @property
    def dataset_config(self) -> DatasetConfig:
        return self.config.dataset_config
