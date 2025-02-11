from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from callbacks.tensorboard_logger import TensorboardLogger
from metrics import ObjectDetectionMAPWrapper
from utils.state import DataKey, EvaluatorState, StoreKey, TrainerState


class ObjectDetectionMAPLogger(TensorboardLogger):
    def __init__(self, metric: ObjectDetectionMAPWrapper, **kwargs):
        """
        Callback for computing and logging a metric.
        """
        super().__init__(**kwargs)
        self.output_dir = Path(self.output_dir) / 'metrics'
        os.makedirs(self.output_dir, exist_ok = True)

        self.metric = metric
        self.history: dict[str, list[float]] = defaultdict(list)

        self.is_main_worker = self.rank in {-1, 0}

    def _update_metric(self, state: TrainerState | EvaluatorState):
        self.metric.update(
            outputs=state.store[StoreKey.OUTPUT],
            targets=state.store[StoreKey.DATA][DataKey.LABEL],
            image_size=state.store[StoreKey.DATA][DataKey.IMAGE].shape[-2:]
        )

    def _update_history(self, mode, values):
        if not self.is_main_worker:
            return

        for name, value in values.items():
            value = value.item() if isinstance(value, torch.Tensor) else value
            self.history[f'{mode}/{name}'].append(value)
            self.log_scalar(f'metrics/{mode}/{name}', value, log_to_console=True)

    def on_train_batch_end(self, state: TrainerState) -> None:
        self._update_metric(state)

    def on_evaluation_batch_end(self, state: EvaluatorState) -> None:
        self._update_metric(state)

    def on_epoch_end(self, state: TrainerState):
        train_values = self.metric.compute()
        self.metric.reset()

        self._update_history('Training', train_values)

    def on_evaluation_end(self, state: EvaluatorState) -> None:
        val_values = self.metric.compute()
        self.metric.reset()

        self._update_history('Validation', val_values)

        if self.is_main_worker:
            pd.DataFrame(self.history).to_csv(self.output_dir / 'map.csv')
