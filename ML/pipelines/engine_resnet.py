from models import SingleTaskModel, ResnetBackbone, DETR, Transformer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from loops import Trainer, Evaluator
from torch.optim import AdamW
from .base import BaseRunner


class TrainRunnerResnet(BaseRunner):
    def __init__(self, distributed, output_dir):
        super().__init__(distributed, output_dir)
        backbone_resnet = ResnetBackbone(
            type_backbone="resnet50",
            hidden_dim=384,
            type_embedding="learned",
            train_backbone=False,
            return_interm_layers=False
        )
        detr_transformer = Transformer(d_model=384,
                                       dropout=0.2,
                                       nhead=6,
                                       dim_feedforward=2048,
                                       num_decoder_layers=6,
                                       num_encoder_layers=6,
                                       normalize_before=True,
                                       return_intermediate_dec=False)
        head = DETR(
            num_channels=2048,
            transformer=detr_transformer,
            num_queries=40,
            aux_loss=False,
            resnet_backbone=True,
        )
        full_model = SingleTaskModel(
            backbone=backbone_resnet, head=head, freeze_backbone=True)
        full_model.load_state_dict(torch.load(
            "weights/full_model_weights.pth", map_location=self.device))
        if distributed:
            full_model = DDP(full_model, device_ids=[
                                self.device], find_unused_parameters=True)
        
        self.optimizer = AdamW(full_model.parameters(), lr=1e-4, weight_decay=1e-4)
        evaluator = Evaluator(self.dataloader_val, loss=self.loss, callbacks=self.callbacks)
        self.trainer = Trainer(dataset=self.dataloader_train, model=full_model, optimizer=self.optimizer, loss=self.loss, evaluator=evaluator, callbacks=self.callbacks)

    def run(self):
        self.trainer.train()


if __name__ == "__main__":
    full_trainer = TrainRunnerResnet(distributed=True)
    full_trainer()
