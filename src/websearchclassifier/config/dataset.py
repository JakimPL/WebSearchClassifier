from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_serializer


class DatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    path: Path
    extension: str
    prompt_column: str
    label_column: str
    confidence_column: Optional[str] = None

    @field_serializer("path")
    def serialize_path(self, path: Path) -> str:
        return str(path)
