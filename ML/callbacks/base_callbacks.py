from __future__ import annotations

import os
from pathlib import Path

from utils import EvaluatorState, TrainerState


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


class CallbackDecorator(Callback):
    """
    Decorator class for callbacks
    """
    def __init__(
        self,
        base_callback: Callback,
    ):
        """
        Initialize the wrapped callback.

        Args:
            base_callback: callback to be wrapped
        """
        self.base_callback = base_callback


class CallbackBatchWithFrequencyDecorator(CallbackDecorator):
    """
    Base class for callbacks that are called during training and evaluation with a given frequency.
    """

    def __init__(
        self, *,
        on_train_batch_frequency: int = 1,
        on_evaluation_batch_frequency: int = 1,
        base_callback: Callback,
    ):
        """
        Set calling frequencies for the *_batch_* methods to the requested frequency.

        Args:
            on_train_batch_frequency: How many iterations are between two calls of the on_train_batch_* methods
            on_evaluation_batch_frequency: How many iterations are between two calls of the 
                                           on_evaluation_batch_* methods
            base_callback: callback to be wrapped
        """
        super().__init__(base_callback)
        self.on_train_batch_frequency = on_train_batch_frequency
        self.on_evaluation_batch_frequency = on_evaluation_batch_frequency

    def on_train_batch_begin(self, state: TrainerState) -> None:
        if state.iteration % self.on_train_batch_frequency == 0:
            self.base_callback.on_train_batch_begin(state)

    def on_train_batch_end(self, state: TrainerState) -> None:
        if state.iteration % self.on_train_batch_frequency == 0:
            self.base_callback.on_train_batch_end(state)

    def on_evaluation_batch_begin(self, state: EvaluatorState) -> None:
        if state.iteration % self.on_evaluation_batch_frequency == 0:
            self.base_callback.on_evaluation_batch_begin(state)

    def on_evaluation_batch_end(self, state: EvaluatorState) -> None:
        if state.iteration % self.on_evaluation_batch_frequency == 0:
            self.base_callback.on_evaluation_batch_end(state)


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
