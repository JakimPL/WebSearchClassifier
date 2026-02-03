from typing import TypeVar

from websearchclassifier.config.model.base import SearchClassifierConfig

ClassifierConfigT = TypeVar("ClassifierConfigT", bound=SearchClassifierConfig)
ConfigT = TypeVar("ConfigT", bound=SearchClassifierConfig)
