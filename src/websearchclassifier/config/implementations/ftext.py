from pathlib import Path

from pydantic import field_serializer

from websearchclassifier.config.base import SearchClassifierConfig


class FastTextSearchClassifierConfig(SearchClassifierConfig):
    name: str = "fasttext"
    embeddig_dim: int = 300
    classifier_type: str = "logistic"
    regularization_strength: float = 1.0
    embeddings_path: Path = Path("cc.en.300.bin")
    random_state: int = 137

    @field_serializer("embeddings_path")
    def serialize_embeddings_path(self, path: Path) -> str:
        return str(path)
