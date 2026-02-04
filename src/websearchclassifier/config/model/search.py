from pydantic import BaseModel, ConfigDict

from websearchclassifier.config.types import BaselineConfigUnion, ClassifierConfigUnion


class WebSearchClassifierConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        use_enum_values=True,
    )

    baseline: BaselineConfigUnion
    classifier: ClassifierConfigUnion
    use_class_weights: bool = True
