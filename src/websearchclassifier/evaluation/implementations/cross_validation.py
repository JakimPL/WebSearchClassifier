from typing import Any, List, Tuple

import numpy as np

from websearchclassifier.config import CrossValidationEvaluatorConfig
from websearchclassifier.dataset import Dataset
from websearchclassifier.evaluation.base import Evaluator
from websearchclassifier.evaluation.types import Metric
from websearchclassifier.model import SearchClassifier


class CrossValidationEvaluator(Evaluator[CrossValidationEvaluatorConfig]):
    def __call__(
        self,
        model: SearchClassifier[Any],
        dataset: Dataset,
        metric: Metric,
        **init_kwargs: Any,
    ) -> float:
        """
        Perform cross-validation evaluation on the given dataset using the specified metric.

        Args:
            model (SearchClassifier): The model to evaluate.
            dataset (Dataset): The dataset to evaluate.
            metric (Metric): The metric to use for evaluation.
            **init_kwargs: Additional keyword arguments for model initialization.

        Returns:
            float: The evaluation score.
        """
        scores: List[float] = []
        config = model.config
        for fold in range(self.config.folds):
            train_data, test_data = self._cross_validation_split(dataset, fold)

            fold_model = model.__class__(config, **init_kwargs)
            fold_model.fit(train_data)
            predictions = fold_model.predict_proba(test_data.prompts)

            score = float(metric(test_data.labels, predictions))
            scores.append(score)

        return float(np.mean(scores))

    def _cross_validation_split(self, dataset: Dataset, fold: int) -> Tuple[Dataset, Dataset]:
        """
        Split the dataset into training and testing sets for the given fold.

        Args:
            dataset (Dataset): The dataset to split.
            fold (int): The current fold index.

        Returns:
            Tuple[Dataset, Dataset]: A tuple containing the training and testing datasets.
        """
        shuffled_dataset = dataset.shuffle(random_seed=self.config.random_seed)
        fold_size = len(shuffled_dataset) // self.config.folds
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold != self.config.folds - 1 else len(shuffled_dataset)

        test_dataset = shuffled_dataset[start_idx:end_idx]
        train_dataset = Dataset.concatenate([shuffled_dataset[:start_idx], shuffled_dataset[end_idx:]])
        return train_dataset, test_dataset
