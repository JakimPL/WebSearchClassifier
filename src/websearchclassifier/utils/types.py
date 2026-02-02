from pathlib import Path
from typing import Any, TypeAlias, Union, get_args

import numpy as np

Bool: TypeAlias = Union[bool, np.bool_]
Integer: TypeAlias = Union[int, np.integer]
Float: TypeAlias = Union[float, np.floating]
String: TypeAlias = Union[str, np.str_]
Pathlike: TypeAlias = Union[str, Path]


def does_belong_to_union(value: Any, union_type: Any) -> bool:
    return isinstance(value, get_args(union_type))


def is_bool(value: Any) -> bool:
    return does_belong_to_union(value, Bool)


def is_integer(value: Any) -> bool:
    return does_belong_to_union(value, Integer)


def is_float(value: Any) -> bool:
    return does_belong_to_union(value, Float)


def is_string(value: Any) -> bool:
    return does_belong_to_union(value, String)
