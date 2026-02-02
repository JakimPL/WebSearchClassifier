from pydantic import BaseModel, ConfigDict

from websearchclassifier.config.dataset.dataset import DatasetConfig


class EvaluatorConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    dataset_config: DatasetConfig
