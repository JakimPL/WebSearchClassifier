from enum import StrEnum
from pathlib import Path

import pandas as pd


class DatasetFormat(StrEnum):
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "xlsx"

    def load(self, path: Path) -> pd.DataFrame:
        match self:
            case DatasetFormat.CSV:
                return pd.read_csv(path)
            case DatasetFormat.PARQUET:
                return pd.read_parquet(path)
            case DatasetFormat.EXCEL:
                return pd.read_excel(path)

        raise ValueError(f"Unsupported dataset format: {self}")
