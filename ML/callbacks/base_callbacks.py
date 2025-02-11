from __future__ import annotations

import os
from pathlib import Path

from utils.state import EvaluatorState, TrainerState


class Callback:
    """Base class for callbacks that are called during training and evaluation."""

    def on_training_begin(self, state: TrainerState) -> None:
        pass

    def on_training_end(self, state: TrainerState) -> None:
        pass

    def on_epoch_begin(self, state: TrainerState) -> None:
        pass

    def on_epoch_end(self, state: TrainerState) -> None:
        pass

    def on_train_batch_begin(self, state: TrainerState) -> None:
        pass

    def on_train_batch_end(self, state: TrainerState) -> None:
        pass

    def on_evaluation_begin(self, state: EvaluatorState) -> None:
        pass

    def on_evaluation_end(self, state: EvaluatorState) -> None:
        pass

    def on_evaluation_batch_begin(self, state: EvaluatorState) -> None:
        pass

    def on_evaluation_batch_end(self, state: EvaluatorState) -> None:
        pass


class CallbackWithOutput(Callback):
    """
    Base class for callbacks which save something.
    """
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)


class CallbackCollection(Callback):
    """Calls callback methods of all callbacks in the collection."""

    def __init__(self, callbacks: list[Callback]):
        self.callbacks = callbacks

    def on_training_begin(self, state: TrainerState) -> None:
        for callback in self.callbacks:
            callback.on_training_begin(state)

    def on_training_end(self, state: TrainerState) -> None:
        for callback in self.callbacks:
            callback.on_training_end(state)

    def on_epoch_begin(self, state: TrainerState) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(state)

    def on_epoch_end(self, state: TrainerState) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(state)

    def on_train_batch_begin(self, state: TrainerState) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_begin(state)

    def on_train_batch_end(self, state: TrainerState) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_end(state)

    def on_evaluation_begin(self, state: EvaluatorState) -> None:
        for callback in self.callbacks:
            callback.on_evaluation_begin(state)

    def on_evaluation_end(self, state: EvaluatorState) -> None:
        for callback in self.callbacks:
            callback.on_evaluation_end(state)

    def on_evaluation_batch_begin(self, state: EvaluatorState) -> None:
        for callback in self.callbacks:
            callback.on_evaluation_batch_begin(state)

    def on_evaluation_batch_end(self, state: EvaluatorState) -> None:
        for callback in self.callbacks:
            callback.on_evaluation_batch_end(state)
