from typing import Literal, TypeAlias, Union

from websearchclassifier.pipeline.models import ModelType

ModelTypeLiteral = Literal["tfidf", "fasttext"]
ModelTypeLike: TypeAlias = Union[ModelTypeLiteral, ModelType]
