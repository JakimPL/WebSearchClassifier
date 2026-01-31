from pydantic import BaseModel, ConfigDict


class SearchClassifierConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    name: str
