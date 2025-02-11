import logging

from callbacks.base_callbacks import Callback
from utils import TrainerState

logger = logging.getLogger(__name__)

class EpochLogger(Callback):
    def on_epoch_begin(self, state: TrainerState):
        logger.info("Epoch %s", state.epoch)
