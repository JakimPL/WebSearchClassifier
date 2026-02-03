from typing import Union

from pydantic import BaseModel, ConfigDict

from websearchclassifier.config.classifier.implementations.logistic import LogisticRegressionConfig
from websearchclassifier.config.classifier.implementations.svm import SVMConfig
from websearchclassifier.config.model.type import ModelType


class SearchClassifierConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        use_enum_values=True,
    )

    type: ModelType
    classifier_config: Union[LogisticRegressionConfig, SVMConfig]
    use_class_weights: bool = True
