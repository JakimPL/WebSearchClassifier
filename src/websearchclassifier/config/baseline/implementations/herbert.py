from websearchclassifier.config.baseline.base import BaselineConfig
from websearchclassifier.config.baseline.type import BaselineType
from websearchclassifier.utils import Device


class HerBERTConfig(BaselineConfig):
    type: BaselineType = BaselineType.HERBERT
    model_name: str = "allegro/herbert-base-cased"
    batch_size: int = 32
    max_length: int = 128
    device: Device = Device.AUTO
