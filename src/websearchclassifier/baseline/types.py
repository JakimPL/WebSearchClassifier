from typing import Any, TypeVar

from websearchclassifier.baseline.base import Baseline

BaselineT = TypeVar("BaselineT", bound=Baseline[Any])
