from __future__ import annotations

from functools import cached_property
from typing import List, Optional, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from websearchclassifier.config import DatasetConfig
from websearchclassifier.dataset.format import DatasetFormat
from websearchclassifier.dataset.item import DatasetItem


class Dataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    config: DatasetConfig
    prompts: np.ndarray
    labels: np.ndarray
    confidence: Optional[np.ndarray] = None

    @cached_property
    def data(self) -> List[DatasetItem]:
        return [
            DatasetItem(prompt, label, confidence)
            for prompt, label, confidence in zip(
                self.prompts,
                self.labels,
                self.confidence if self.confidence is not None else [None] * len(self.prompts),
            )
        ]

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
        dataset_format = DatasetFormat(config.extension.lower())
        dataframe = dataset_format.load(config.path)
        cls._validate(config, dataframe)
        return cls(
            config=config,
            prompts=dataframe[config.prompt_column].to_numpy(),
            labels=dataframe[config.label_column].to_numpy(dtype=bool),
            confidence=dataframe[config.confidence_column].to_numpy(dtype=float) if config.confidence_column else None,
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
