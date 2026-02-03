from websearchclassifier.utils.device import Device
from websearchclassifier.utils.logger import setup_logger
from websearchclassifier.utils.types import (
    Bool,
    ClassifierT_co,
    Float,
    Integer,
    Pathlike,
    ProbabilisticClassifier,
    ProbabilisticClassifierT_co,
    String,
    TorchModuleT,
    Weights,
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
    "TorchModuleT",
    "ClassifierT_co",
    "ProbabilisticClassifier",
    "ProbabilisticClassifierT_co",
    "Device",
    "Bool",
    "Integer",
    "Float",
    "String",
    "Pathlike",
    "Weights",
    "does_belong_to_union",
    "is_bool",
    "is_integer",
    "is_float",
    "is_string",
]
