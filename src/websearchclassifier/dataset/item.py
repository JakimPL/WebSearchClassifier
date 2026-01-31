from typing import NamedTuple, Optional


class DatasetItem(NamedTuple):
    prompt: str
    search: bool
    confidence: Optional[float] = None
