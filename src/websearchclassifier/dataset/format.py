from enum import StrEnum

import pandas as pd

from websearchclassifier.config.dataset.dataset import DatasetConfig


class DatasetFormat(StrEnum):
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "xlsx"

    def load(self, config: DatasetConfig) -> pd.DataFrame:
        path = config.path
        decimal = config.decimal_separator
        if path is None:
            raise ValueError("DatasetConfig.path must be provided to load a dataset from file")

        match self:
            case DatasetFormat.CSV:
                return pd.read_csv(path, decimal=decimal)
            case DatasetFormat.PARQUET:
                return pd.read_parquet(path, decimal=decimal)
            case DatasetFormat.EXCEL:
                return pd.read_excel(path, decimal=decimal)

        raise ValueError(f"Unsupported dataset format: {self}")
