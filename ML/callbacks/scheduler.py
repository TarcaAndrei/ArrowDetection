from __future__ import annotations

import logging
from functools import partial

import torch
from torch.optim.lr_scheduler import LRScheduler

from callbacks.base_callbacks import Callback
from utils import TrainerState

logger = logging.getLogger(__name__)

class TorchScheduler(Callback):
    def __init__(self, scheduler_class: str, **kwargs):
        """
        Wrapper for torch schedulers.

        Args:
            scheduler_class: scheduler class
            kwargs: scheduler parameters
        """
        self.scheduler_initializer = partial(getattr(torch.optim.lr_scheduler, scheduler_class), **kwargs)
        self.scheduler: LRScheduler | None = None

    def on_training_begin(self, state: TrainerState) -> None:
        self.scheduler = self.scheduler_initializer(optimizer=state.optimizer)

    def on_epoch_end(self, state: TrainerState) -> None:
        if self.scheduler is None:
            raise ValueError("The scheduler is not initialized.")
        self.scheduler.step()
        logger.info("Update lr, the new value is: %f", self.scheduler.get_last_lr()[0])
