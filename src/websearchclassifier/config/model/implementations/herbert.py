from websearchclassifier.config.model.base import SearchClassifierConfig
from websearchclassifier.config.model.type import ModelType
from websearchclassifier.utils import Device


class HerBERTSearchClassifierConfig(SearchClassifierConfig):
    type: ModelType = ModelType.HERBERT
    model_name: str = "allegro/herbert-base-cased"
    batch_size: int = 32
    max_length: int = 128
    device: Device = Device.AUTO
