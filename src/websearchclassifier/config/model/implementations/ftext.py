from pathlib import Path
from typing import Literal

from pydantic import field_serializer

from websearchclassifier.config.model.base import SearchClassifierConfig


class FastTextSearchClassifierConfig(SearchClassifierConfig):
    name: str = "fasttext"
    embedding_dim: int = 300
    classifier_type: Literal["logistic", "svm"] = "logistic"
    regularization_strength: float = 1.0
    embeddings_path: Path = Path("cc.en.300.bin")
    random_state: int = 137

    @field_serializer("embeddings_path")
    def serialize_embeddings_path(self, path: Path) -> str:
        return str(path)
