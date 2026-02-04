import pickle
from typing import Any

from websearchclassifier.utils.types import Pathlike


def save_pickle(data: Any, filepath: Pathlike) -> None:
    with open(filepath, "wb") as file:
        pickle.dump(data, file)


def load_pickle(filepath: Pathlike) -> Any:
    with open(filepath, "rb") as file:
        return pickle.load(file)
