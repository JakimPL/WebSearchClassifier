from pathlib import Path

from pydantic import BaseModel, field_serializer, field_validator


class DatasetPathConfig(BaseModel):
    path: Path

    @field_validator("path", mode="before")
    def validate_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_serializer("path")
    def serialize_path(self, path: Path) -> str:
        return str(path)

    @property
    def extension(self) -> str:
        return self.path.suffix.lower().lstrip(".")

    @property
    def filename(self) -> str:
        return self.path.name
