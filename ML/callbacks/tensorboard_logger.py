from __future__ import annotations

import logging
import os
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

from callbacks.base_callbacks import CallbackWithOutput
from utils import TrainerState

logger = logging.getLogger(__name__)


class TensorboardLogger(CallbackWithOutput):
    def __init__(self, **kwargs):
        """
        Callback for logging values to tensorboard.
        """
        super().__init__(**kwargs)
        tensorboard_dir = self.output_dir / "tensorboard"
        os.makedirs(tensorboard_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        self.iterations: dict = defaultdict(int)
        self.rank = int(os.getenv("LOCAL_RANK", "-1"))

    def log_scalar(self, name: str, number: int | float, log_to_console=False):
        """
        Logs a single number
        """
        if self.rank != -1:
            name = f'rank_{self.rank}/{name}'
        self.writer.add_scalar(name, number, self.iterations[name])
        self.iterations[name] += 1
        if log_to_console:
            logger.info("%s: %f", name, number)

    def log_scalars(self, name: str, numbers: dict, log_to_console=False):
        """
        Logs a dictionary with numbers which will appear on the same plot
        """
        if self.rank != -1:
            name = f'rank_{self.rank}/{name}'
        self.writer.add_scalars(name, numbers, self.iterations[name])
        self.iterations[name] += 1
        if log_to_console:
            for sub_name, number in numbers.items():
                logger.info("%s: %f", f"{name}/{sub_name}", number)

    def on_training_end(self, state: TrainerState) -> None:
        self.writer.close()
