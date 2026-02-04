from pathlib import Path

from pydantic import field_serializer

from websearchclassifier.config.baseline.base import BaselineConfig
from websearchclassifier.config.baseline.type import BaselineType


class FastTextConfig(BaselineConfig):
    type: BaselineType = BaselineType.FASTTEXT
    embedding_dim: int = 300
    embeddings_path: Path = Path("cc.en.300.bin")

    @field_serializer("embeddings_path")
    def serialize_embeddings_path(self, path: Path) -> str:
        return str(path)
