from typing import NamedTuple, Optional

from websearchclassifier.dataset.types import Label, Prediction, Prompt


class DatasetItem(NamedTuple):
    prompt: Prompt
    search: Label
    confidence: Optional[Prediction] = None
