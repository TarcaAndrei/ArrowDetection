import logging
import os
from pathlib import Path

import torch

from callbacks.base_callbacks import CallbackWithOutput
from utils.state import TrainerState

logger = logging.getLogger(__name__)

class ModelSaver(CallbackWithOutput):
    def __init__(self, *, save_steps: int = -1, **kwargs):
        """
        Callback used for saving the model.

        Args:
            save_steps: number of epochs between two checkpoint saves; if -1, it saves only the final state
        """
        super().__init__(**kwargs)
        self.output_dir = Path(self.output_dir) / 'checkpoints'
        os.makedirs(self.output_dir, exist_ok=True)

        self.save_steps = save_steps

    def _should_save(self, epochs: int) -> bool:
        """
        Function to check if the trainer should save the model or not
        Args:
            epochs: the current number of epochs (or -1 if the training has ended)
        Returns:
            True if the model should be saved 
            False otherwise
        """
        # if the save_steps are not set and the training hasn't finished yet
        if epochs == -1:
            return True

        if self.save_steps == -1:
            return False

        # if the step is ok
        if epochs % self.save_steps != 0:
            return False

        return True

    def _save(self, file_path, model, epoch):
        """Helper method to save a model at a given epoch to a given path."""
        if self._should_save(epoch):
            logger.info("Saving the model to  %s", file_path)
            if os.getenv("LOCAL_RANK") is None:
                # non distributed setup
                torch.save(model.state_dict(), file_path)
            else:
                torch.save(model.module.state_dict(), file_path)

    def on_epoch_end(self, state: TrainerState) -> None:
        file_path = self.output_dir / f"model_{state.epoch}.pth"
        self._save(file_path, state.model, state.epoch + 1)

    def on_training_end(self, state: TrainerState) -> None:
        file_path = self.output_dir / "model_final.pth"
        self._save(file_path, state.model, -1)
