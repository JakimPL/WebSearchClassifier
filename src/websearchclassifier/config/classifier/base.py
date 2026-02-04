from abc import ABC

from pydantic import BaseModel, ConfigDict

from websearchclassifier.config.classifier.type import ClassifierType


class ClassifierConfig(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        use_enum_values=True,
    )

    type: ClassifierType
    random_state: int = 137
