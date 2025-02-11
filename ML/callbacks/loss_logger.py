from __future__ import annotations

from collections import defaultdict

import numpy as np

from callbacks.tensorboard_logger import TensorboardLogger
from utils import EvaluatorState, StoreKey, TrainerState


class LossLogger(TensorboardLogger):
    def __init__(self, **kwargs):
        """
        Callback for computing and logging a metric.
        """
        super().__init__(**kwargs)
        self.train_epoch_losses = defaultdict(list)
        self.eval_epoch_losses = defaultdict(list)
        self.lrs = []

    def on_train_batch_end(self, state: TrainerState) -> None:
        losses = state.store[StoreKey.LOSSES]
        lr = state.optimizer.param_groups[0]['lr']

        self.lrs.append(lr)
        self.log_scalar('batch/train/lr', lr)

        for name, value in losses.items():
            self.log_scalar(f'batch/train/{name}', value, log_to_console=state.iteration==0)
            self.train_epoch_losses[name].append(value)

    def on_evaluation_batch_end(self, state: EvaluatorState) -> None:
        losses = state.store[StoreKey.LOSSES]

        for name, value in losses.items():
            self.log_scalar(f'batch/eval/{name}', value, log_to_console=state.iteration==0)
            self.eval_epoch_losses[name].append(value)

    def on_evaluation_end(self, state: EvaluatorState) -> None:
        if len(self.lrs) > 0:
            self.log_scalar('epoch/lr', np.mean(self.lrs))

        if len(self.train_epoch_losses) > 0 and len(self.eval_epoch_losses) > 0:
            # log both eval and train mean losses over the epoch
            for name, train_values in self.train_epoch_losses.items():
                mean_train = np.mean(train_values)
                mean_eval = np.mean(self.eval_epoch_losses[name])

                self.log_scalars(f'epoch/{name}', {
                        'train': mean_train,
                        'eval': mean_eval,
                    }, log_to_console=True
                )
        else:
            # log either train or eval mean loss over the epoch
            losses_dict = {}
            if len(self.train_epoch_losses) > 0:
                losses_dict = self.train_epoch_losses
                stage = 'train'
            elif len(self.eval_epoch_losses) > 0:
                losses_dict = self.eval_epoch_losses
                stage = 'eval'

            for name, values in losses_dict.items():
                self.log_scalar(f'epoch/{stage}/{name}', float(np.mean(values)))

        # reset history
        self.train_epoch_losses = defaultdict(list)
        self.eval_epoch_losses = defaultdict(list)
        self.lrs = []
