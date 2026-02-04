from __future__ import annotations

from functools import cached_property
from typing import List, Optional, Self, Sequence, Tuple, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.utils import compute_class_weight

from websearchclassifier.config import DatasetConfig, WeightingScheme
from websearchclassifier.dataset.format import DatasetFormat
from websearchclassifier.dataset.item import DatasetItem
from websearchclassifier.dataset.types import (
    Label,
    Labels,
    Prediction,
    Predictions,
    Prompts,
    is_label,
    is_prediction,
    is_prompt,
)
from websearchclassifier.utils import String, Weights


class Dataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    config: DatasetConfig = Field(default_factory=DatasetConfig)
    prompts: npt.NDArray[np.str_]
    labels: npt.NDArray[np.bool_]
    confidence: Optional[npt.NDArray[np.floating]] = None

    @model_validator(mode="after")
    def _validate_lengths(self) -> Dataset:
        if len(self.prompts) != len(self.labels):
            raise ValueError("prompts and labels must have the same length")

        if self.confidence is not None and len(self.prompts) != len(self.confidence):
            raise ValueError("prompts and confidence must have the same length")

        self.prompts = self.prompts.astype(np.str_)
        self.labels = self.labels.astype(np.bool_)
        if self.confidence is not None:
            self.confidence = self.confidence.astype(np.float16)

        return self

    def __getitem__(self, index: Union[int, slice, np.ndarray]) -> Self:
        return self.__class__(
            config=self.config,
            prompts=self.prompts[index],
            labels=self.labels[index],
            confidence=self.confidence[index] if self.confidence is not None else None,
        )

    def __len__(self) -> int:
        return len(self.prompts)

    @cached_property
    def items(self) -> List[DatasetItem]:
        return [
            DatasetItem(prompt, label, confidence)
            for prompt, label, confidence in zip(
                self.prompts,
                self.labels,
                self.confidence if self.confidence is not None else [None] * len(self.prompts),
            )
        ]

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        data = {
            self.config.prompt_column: self.prompts,
            self.config.label_column: self.labels,
        }

        if self.confidence is not None:
            data[self.config.confidence_column] = self.confidence

        return pd.DataFrame(data, columns=list(data.keys()))

    @cached_property
    def size(self) -> int:
        return len(self.prompts)

    @cached_property
    def positive(self) -> int:
        return int(np.sum(self.labels))

    @cached_property
    def negative(self) -> int:
        return self.size - self.positive

    @classmethod
    def load(cls, config: DatasetConfig) -> Self:
        if config.dataset_path is None:
            raise ValueError("DatasetConfig.path must be provided to load a dataset from file")

        assert config.extension is not None
        dataset_format = DatasetFormat(config.extension)
        dataframe = dataset_format.load(config)
        cls._validate(config, dataframe)
        return cls(
            config=config,
            prompts=dataframe[config.prompt_column].to_numpy(dtype=np.str_),
            labels=dataframe[config.label_column].to_numpy(dtype=np.bool_),
            confidence=(
                dataframe[config.confidence_column].to_numpy(dtype=np.float16)
                if config.confidence_column in dataframe.columns
                else None
            ),
        )

    @staticmethod
    def _validate(config: DatasetConfig, dataframe: pd.DataFrame) -> None:
        missing_columns = [
            column
            for column in [config.prompt_column, config.label_column, config.confidence_column]
            if column is not None and column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns in dataset: {', '.join(missing_columns)}")

    @classmethod
    def concatenate(cls, datasets: Sequence[Self]) -> Self:
        if not datasets:
            raise ValueError("Cannot concatenate empty list of datasets")

        config = datasets[0].config
        for dataset in datasets[1:]:
            if dataset.config != config:
                raise ValueError("All datasets must have the same config to be concatenated")

        return cls(
            config=config,
            prompts=np.concatenate([dataset.prompts for dataset in datasets]),
            labels=np.concatenate([dataset.labels for dataset in datasets]),
            confidence=(
                np.concatenate([dataset.confidence for dataset in datasets if dataset.confidence is not None])
                if all(dataset.confidence is not None for dataset in datasets)
                else None
            ),
        )

    @classmethod
    def _split_regular(
        cls,
        dataset: Self,
        fraction: float,
        random_seed: Optional[int] = None,
    ) -> Tuple[Self, Self]:
        shuffled_dataset = dataset.shuffle(random_seed=random_seed)
        split_idx = int(shuffled_dataset.size * fraction)
        train_dataset = cls(
            config=shuffled_dataset.config,
            prompts=shuffled_dataset.prompts[:split_idx],
            labels=shuffled_dataset.labels[:split_idx],
            confidence=shuffled_dataset.confidence[:split_idx] if shuffled_dataset.confidence is not None else None,
        )

        test_dataset = cls(
            config=shuffled_dataset.config,
            prompts=shuffled_dataset.prompts[split_idx:],
            labels=shuffled_dataset.labels[split_idx:],
            confidence=shuffled_dataset.confidence[split_idx:] if shuffled_dataset.confidence is not None else None,
        )

        return train_dataset, test_dataset

    @classmethod
    def _split_stratified(cls, dataset: Self, fraction: float, random_seed: Optional[int] = None) -> Tuple[Self, Self]:
        positive_indices = np.where(dataset.labels)[0]
        negative_indices = np.where(~dataset.labels)[0]
        train_positive_dataset, test_positive_dataset = cls._split_regular(
            dataset[positive_indices],
            fraction,
            random_seed,
        )
        train_negative_dataset, test_negative_dataset = cls._split_regular(
            dataset[negative_indices],
            fraction,
            random_seed,
        )

        train_dataset = cls.concatenate([train_positive_dataset, train_negative_dataset])
        test_dataset = cls.concatenate([test_positive_dataset, test_negative_dataset])
        return train_dataset, test_dataset

    def split(
        self,
        fraction: float,
        stratify: bool = True,
        random_seed: Optional[int] = None,
    ) -> Tuple[Self, Self]:
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must be between 0 and 1, got {fraction}")

        if stratify:
            return self._split_stratified(self, fraction, random_seed)

        return self._split_regular(self, fraction, random_seed)

    def shuffle(self, random_seed: Optional[int] = None) -> Self:
        np.random.seed(random_seed)
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        return self[indices]

    @classmethod
    def from_dataset_like(
        cls,
        dataset: Optional[Union[Dataset, pd.DataFrame]],
        config: Optional[DatasetConfig] = None,
    ) -> Self:
        """
        Creates a Dataset from a dataset-like object.

        Args:
            dataset: Dataset-like object.
            config: Dataset configuration.

        Returns:
            Dataset object.
        """
        if isinstance(dataset, cls):
            return dataset

        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("dataset must be a pandas DataFrame or a Dataset instance")

        if config is None:
            raise ValueError("config must be provided when creating Dataset from DataFrame")

        return cls(
            config=config,
            prompts=dataset[config.prompt_column].to_numpy(dtype=np.str_),
            labels=dataset[config.label_column].to_numpy(dtype=np.bool_),
            confidence=(
                dataset[config.confidence_column].to_numpy(dtype=np.float16)
                if config.confidence_column in dataset.columns
                else None
            ),
        )

    @classmethod
    def create(
        cls,
        dataset: Optional[Union[Dataset, pd.DataFrame]] = None,
        prompts: Optional[Union[String, Prompts]] = None,
        labels: Optional[Union[Label, Labels]] = None,
        confidence: Optional[Union[Prediction, Predictions]] = None,
        config: Optional[DatasetConfig] = None,
    ) -> Self:
        """
        Creates a dataset out of prompts and labels or a dataset-like object.

        Accepts either prompts and labels directly, or a dataset object.

        Args:
            dataset: Dataset object.
            prompts: Single prompt or list of prompts.
            labels: Single label or list of labels.
            config: Dataset configuration for dataset-like input.

        Returns:
            Dataset object.
        """
        if prompts is None and labels is None and dataset is None:
            raise ValueError("At least one of prompts/labels or dataset must be provided.")

        if dataset is not None:
            if prompts is not None or labels is not None:
                raise ValueError("If dataset is provided, prompts and labels must be None.")

            return cls.from_dataset_like(dataset, config=config)

        if prompts is None or labels is None:
            raise ValueError("Both prompts and labels must be provided when dataset is not given.")

        prompts = cls.normalize_prompts(prompts)
        labels = cls.normalize_labels(labels)
        confidence = cls.normalize_confidence(confidence)

        return cls(
            config=config if config is not None else DatasetConfig(),
            prompts=prompts,
            labels=labels,
            confidence=confidence,
        )

    @staticmethod
    def normalize_prompts(prompts: Union[String, Prompts]) -> npt.NDArray[np.str_]:
        return np.array([prompts] if is_prompt(prompts) else prompts, dtype=np.str_)

    @staticmethod
    def normalize_labels(labels: Union[Label, Labels]) -> npt.NDArray[np.bool_]:
        return np.array([labels] if is_label(labels) else labels, dtype=np.bool_)

    @staticmethod
    def normalize_confidence(
        confidence: Optional[Union[Prediction, Predictions]],
    ) -> Optional[npt.NDArray[np.floating]]:
        if confidence is None:
            return None

        return np.array([confidence] if is_prediction(confidence) else confidence, dtype=np.float16)

    def compute_class_weights(self) -> Weights:
        """Compute class weights based on the selected weighting scheme.

        Args:
            labels: A list or array of class labels.

        Returns:
            A dictionary mapping class labels to their computed weights.
        """
        weights: Weights
        classes = np.array([0, 1], dtype=np.int8)
        array: npt.NDArray[np.int8] = np.array(self.labels, dtype=np.int8)
        match self.config.weighting_scheme:
            case WeightingScheme.NONE:
                weights = {label: 1.0 for label in classes}

            case WeightingScheme.BALANCED:
                class_weight = compute_class_weight(
                    class_weight="balanced",
                    classes=classes,
                    y=array.astype(int),
                )
                weights = {0: class_weight[0], 1: class_weight[1]}

            case WeightingScheme.INVERSE:
                counts = np.bincount(array.astype(int))
                total = len(array)
                class_weight = total / (2 * counts)
                weights = {0: class_weight[0], 1: class_weight[1]}

            case _:
                raise ValueError(f"Unknown weighting scheme: {self.config.weighting_scheme}")

        return weights


DatasetLike: TypeAlias = Union[pd.DataFrame, Dataset]
