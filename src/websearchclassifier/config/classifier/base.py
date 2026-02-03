from abc import ABC
from typing import Generic

from pydantic import BaseModel, ConfigDict

from websearchclassifier.config.classifier.type import ClassifierType
from websearchclassifier.utils import ProbabilisticClassifierT_co


class ClassifierConfig(BaseModel, Generic[ProbabilisticClassifierT_co], ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        use_enum_values=True,
    )

    type: ClassifierType
    random_state: int = 137
