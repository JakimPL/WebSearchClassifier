from pathlib import Path
from typing import Optional, Union, cast

from pydantic import BaseModel, ConfigDict, field_validator

from websearchclassifier.config.dataset.path import DatasetPathConfig


class DatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    prompt_column: str = "prompt"
    label_column: str = "label"
    confidence_column: str = "confidence"
    dataset_path: Optional[Union[Path, DatasetPathConfig]] = None
    decimal_separator: str = ","

    @field_validator("dataset_path", mode="before")
    def validate_dataset_path(cls, value: Optional[Union[Path, DatasetPathConfig]]) -> Optional[DatasetPathConfig]:
        if value is None or isinstance(value, DatasetPathConfig):
            return value

        return DatasetPathConfig(path=value)

    @property
    def path(self) -> Optional[Path]:
        if self.dataset_path is None:
            return None

        return cast(DatasetPathConfig, self.dataset_path).path

    @property
    def filename(self) -> Optional[str]:
        if self.dataset_path is None:
            return None

        return cast(DatasetPathConfig, self.dataset_path).filename

    @property
    def extension(self) -> Optional[str]:
        if self.dataset_path is None:
            return None

        return cast(DatasetPathConfig, self.dataset_path).extension
