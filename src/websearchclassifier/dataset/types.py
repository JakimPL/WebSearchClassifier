from typing import Sequence, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from websearchclassifier.utils import Bool, Float, Integer, String, is_bool, is_float, is_integer, is_string

Prompt: TypeAlias = String
Label: TypeAlias = Union[Bool, Integer]
Prediction: TypeAlias = Union[Bool, Integer, Float]

Prompts: TypeAlias = Union[Sequence[str], Sequence[String], npt.NDArray[np.str_], pd.Series]
Labels: TypeAlias = Union[Sequence[bool], Sequence[Label], npt.NDArray[np.bool_], pd.Series]
Predictions: TypeAlias = Union[Sequence[float], Sequence[Prediction], npt.NDArray[np.floating], pd.Series]


def is_prompt(value: object) -> bool:
    return is_string(value)


def is_label(value: object) -> bool:
    if not (value == 0 or value == 1):  # pylint: disable=consider-using-in
        return False

    return is_bool(value) or is_integer(value)


def is_prediction(value: object) -> bool:
    if not (is_bool(value) or is_integer(value) or is_float(value)):
        return False

    assert isinstance(value, np._ConvertibleToFloat)  # pylint: disable=protected-access
    return 0.0 <= float(value) <= 1.0
