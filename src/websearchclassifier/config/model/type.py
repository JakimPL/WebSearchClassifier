from enum import StrEnum
from typing import Literal, TypeAlias, Union


class ModelType(StrEnum):
    TFIDF = "tfidf"
    FASTTEXT = "fasttext"
    HERBERT = "herbert"


ModelTypeLiteral = Literal["tfidf", "fasttext", "herbert"]
ModelTypeLike: TypeAlias = Union[ModelTypeLiteral, ModelType]
