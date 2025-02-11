from __future__ import annotations

import torch

from callbacks import CallbackCollection
from utils import DataKey, StoreKey, TrainerState

from .evaluator import Evaluator


class Trainer:
    def __init__(
        self, *,
        model,
        optimizer: torch.optim.Optimizer,
        loss,
        dataset,
        callbacks: CallbackCollection,
        evaluator: Evaluator,
        epochs: int = 100,
        start_epoch:int = 0,
    ) -> None:
        """
        Contains the training loop logic

        Args:
            model: the network to be trained
            optimizer: the training optimzer
            loss: traning loss
            dataset: data to be used for training
            callbacks: collection of callbacks
            evaluator: performs the validation at the end of the epoch
            epochs: number of epochs to be performed
            start_epoch: starting epoch, defaults to 0, greater than 0 when starting a training based on an another one
        """
        self.model = model
        self.callbacks = callbacks
        self.optimizer = optimizer
        self.loss = loss
        self.dataset = dataset
        self.callbacks = callbacks
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.evaluator = evaluator

        self.state = TrainerState(model, optimizer, start_epoch, 0, {})

    def _train_step(self) -> None:
        """
        Function that performs one step throught the training data
        """
        self.callbacks.on_train_batch_begin(self.state)

        x = self.state.store[StoreKey.DATA][DataKey.IMAGE]
        y = self.state.store[StoreKey.DATA][DataKey.LABEL]
        y_pred = self.model(x)
        loss_dict = self.loss(y_pred, y)
        losses = sum(loss_dict[k] * self.loss.weight_dict[k]
                        for k in loss_dict.keys() if k in self.loss.weight_dict)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        self.state.store.update({
            StoreKey.OUTPUT: y_pred,
            StoreKey.LOSSES: {
                'total_loss': losses.item(),
                **{name: sub_loss.item() for name, sub_loss in loss_dict.items()}
            },
        })

        self.callbacks.on_train_batch_end(self.state)

    def _train_epoch(self) -> None:
        """
        Performs one epoch of training
        """
        self.callbacks.on_epoch_begin(self.state)
        self.model.train()
        for i, data in enumerate(self.dataset):
            self.state.iteration = i
            self.state.store[StoreKey.DATA] = data

            self._train_step()

            self.state.store = {}

        self.callbacks.on_epoch_end(self.state)

    def train(self) -> None:
        """
        The trainining loop
        """
        self.callbacks.on_training_begin(self.state)
        for epoch in range(self.start_epoch, self.epochs):
            self.state.epoch = epoch

            if hasattr(self.dataset.sampler, 'set_epoch'):
                # the distributed sampler need this explicit call in order to shuffle the data
                self.dataset.sampler.set_epoch(epoch)

            self._train_epoch()
            self.evaluator.evaluate(self.model, epoch)

        self.callbacks.on_training_end(self.state)
