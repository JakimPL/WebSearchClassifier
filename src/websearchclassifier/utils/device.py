from __future__ import annotations

from enum import StrEnum
from typing import Dict, Mapping, Union, overload

import torch

from websearchclassifier.utils.types import TorchModuleT


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"

    @classmethod
    def resolve(cls, device: Device) -> Device:
        """
        Resolve AUTO to actual device based on availability.

        Args:
            device: Device enum value to resolve

        Returns:
            Resolved device (CPU or CUDA)
        """
        if device == cls.AUTO:
            return cls.CUDA if torch.cuda.is_available() else cls.CPU

        return device

    @overload
    def to(self, obj: TorchModuleT) -> TorchModuleT: ...

    @overload
    def to(self, obj: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]: ...

    def to(self, obj: Union[TorchModuleT, Mapping[str, torch.Tensor]]) -> Union[TorchModuleT, Dict[str, torch.Tensor]]:
        """
        Move object(s) to this device.

        Supports both PyTorch modules and mappings of tensors (dict, BatchEncoding, etc.).

        Args:
            obj: PyTorch module or mapping of tensors to move

        Returns:
            Object moved to this device
        """
        if isinstance(obj, Mapping):
            return {key: value.to(self.value) for key, value in obj.items()}

        return obj.to(self.value)
