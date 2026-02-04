from typing import Any, Type

from websearchclassifier.baseline.base import Baseline
from websearchclassifier.baseline.implementations.ftext import FastTextBaseline
from websearchclassifier.baseline.implementations.herbert import HerBERTBaseline
from websearchclassifier.baseline.implementations.tfidf import TFIDFBaseline
from websearchclassifier.config import BaselineConfig, BaselineType


def get_baseline_class(baseline_type: BaselineType) -> Type[Baseline[Any]]:
    """
    Factory function to get the baseline class based on the model type.

    Args:
        baseline_type (ModelType): The type of the model.

    Returns:
        Type[Baseline[Any]]: The corresponding baseline class.
    """
    match baseline_type:
        case BaselineType.TFIDF:
            return TFIDFBaseline

        case BaselineType.HERBERT:
            return HerBERTBaseline

        case BaselineType.FASTTEXT:
            return FastTextBaseline

    raise KeyError(f"Unsupported baseline type: {baseline_type}")


def load_baseline(config: BaselineConfig) -> Baseline[Any]:
    """
    Load the baseline corresponding to the given baseline configuration.

    Args:
        config (BaselineConfig): The baseline configuration.

    Returns:
        Baseline[Any]: The corresponding baseline instance.
    """
    BaselineClass = get_baseline_class(config.type)
    baseline: Baseline[Any] = BaselineClass(config)
    return baseline
