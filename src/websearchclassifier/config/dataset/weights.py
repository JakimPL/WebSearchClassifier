from enum import StrEnum
from typing import Literal, TypeAlias, Union


class WeightingScheme(StrEnum):
    NONE = "none"
    BALANCED = "balanced"
    INVERSE = "inverse"


WeightingSchemeLiteral: TypeAlias = Literal["none", "balanced", "inverse"]
WeightingSchemeLike: TypeAlias = Union[WeightingScheme, WeightingSchemeLiteral]
