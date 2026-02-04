from pydantic import BaseModel, ConfigDict

from websearchclassifier.config.baseline.type import BaselineType


class BaselineConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        use_enum_values=True,
    )

    type: BaselineType
