import torch

from callbacks import CallbackCollection
from utils import DataKey, EvaluatorState, StoreKey


class Evaluator:
    def __init__(
        self,
        dataset,
        callbacks: CallbackCollection,
        loss = None
    ) -> None:
        """
        Contains the evaluation loop logic

        Args:
            dataset: data to be used for evaluation
            callbacks: collection of callbacks
            loss: if given will compute the loss during evaluation
        """
        self.dataset = dataset
        self.callbacks = callbacks
        self.loss = loss

        self.state = EvaluatorState(None, 0, 0, {})


    def _eval_step(self) -> None:
        """
        Function that performs one step throught the validation data
        """
        self.callbacks.on_evaluation_batch_begin(self.state)

        x = self.state.store[StoreKey.DATA][DataKey.IMAGE]
        y = self.state.store[StoreKey.DATA][DataKey.LABEL]
        y_pred = self.state.model(x)

        self.state.store[StoreKey.OUTPUT] = y_pred

        if self.loss:
            loss_dict = self.loss(y_pred, y)
            losses = sum(loss_dict[k] * self.loss.weight_dict[k]
                            for k in loss_dict.keys() if k in self.loss.weight_dict)

            self.state.store[StoreKey.LOSSES] = {
                'total_loss': losses.item(),
                **{name: sub_loss.item() for name, sub_loss in loss_dict.items()}
            }

        self.callbacks.on_evaluation_batch_end(self.state)


    def evaluate(self, model, epoch) -> None:
        model.eval()
        self.state = EvaluatorState(model, epoch, 0, {})
        self.callbacks.on_evaluation_begin(self.state)

        with torch.inference_mode():
            for i, data in enumerate(self.dataset):
                self.state.iteration = i
                self.state.store[StoreKey.DATA] = data

                self._eval_step()

                self.state.store = {}

        self.callbacks.on_evaluation_end(self.state)
