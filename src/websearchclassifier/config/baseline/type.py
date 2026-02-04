from enum import StrEnum
from typing import Literal, TypeAlias, Union


class BaselineType(StrEnum):
    TFIDF = "tfidf"
    FASTTEXT = "fasttext"
    HERBERT = "herbert"


BaselineTypeLiteral = Literal["tfidf", "fasttext", "herbert"]
BaselineTypeLike: TypeAlias = Union[BaselineTypeLiteral, BaselineType]
