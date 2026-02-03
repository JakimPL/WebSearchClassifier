from pathlib import Path

from pydantic import field_serializer

from websearchclassifier.config.model.base import SearchClassifierConfig
from websearchclassifier.config.model.type import ModelType


class FastTextSearchClassifierConfig(SearchClassifierConfig):
    type: ModelType = ModelType.FASTTEXT
    embedding_dim: int = 300
    embeddings_path: Path = Path("cc.en.300.bin")

    @field_serializer("embeddings_path")
    def serialize_embeddings_path(self, path: Path) -> str:
        return str(path)
