from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, Dict

import torch


@dataclass
class TrainerState:
    """State object that is passed to the trainer callbacks."""

    model: Any
    optimizer: torch.optim.Optimizer
    epoch: int
    iteration: int
    store: Dict[Any, Any]


@dataclass
class EvaluatorState:
    """State object that is passed to the evaluator callbacks."""

    model: Any
    epoch: int
    iteration: int
    store: Dict[Any, Any]


class StoreKey(IntEnum):
    """Keys that are available to the `store` member of the states."""

    DATA = auto()
    OUTPUT = auto()
    LOSSES = auto()


class DataKey(IntEnum):
    """Keys that are available to the `DATA` member of the store."""

    IMAGE = auto()
    LABEL = auto()
    METADATA = auto()

class LabelType(IntEnum):
    """ Keys that defines the types of labels """
    BBOX = auto()
    KEYPOINT = auto()
