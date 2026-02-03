from pydantic import ConfigDict, Field

from websearchclassifier.config.evaluation.base import EvaluatorConfig


class CrossValidationEvaluatorConfig(EvaluatorConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    folds: int = Field(default=5, ge=2)
    stratify: bool = True
    random_seed: int = 13
