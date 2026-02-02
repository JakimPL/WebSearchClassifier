from websearchclassifier.utils.logger import setup_logger
from websearchclassifier.utils.types import (
    Bool,
    Float,
    Integer,
    Pathlike,
    String,
    does_belong_to_union,
    is_bool,
    is_float,
    is_integer,
    is_string,
)

logger = setup_logger()

__all__ = [
    "setup_logger",
    "logger",
    "Bool",
    "Integer",
    "Float",
    "String",
    "Pathlike",
    "does_belong_to_union",
    "is_bool",
    "is_integer",
    "is_float",
    "is_string",
]
