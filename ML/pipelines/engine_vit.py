import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from .base import BaseRunner
from models import SingleTaskModel, BackboneViT, DETR, Transformer
from loops import Trainer, Evaluator

from torch.optim import AdamW


class TrainRunnerVit(BaseRunner):
    def __init__(self, distributed, output_dir):
        super().__init__(distributed, output_dir)
        backbone_vit = BackboneViT(
            embed_dim=384,
            depth=12,
            num_heads=6,
            patch_tokens=True,
            patch_size=14,
            pretrained_weights='/weights/dinov2_vits14_pretrain_ok.pth',  # vit small
            init_values=1e-5,
            img_size=(518, 1666)
        )
        detr_transformer = Transformer(d_model=384,
                                       dropout=0.2,
                                       nhead=6,
                                       dim_feedforward=2048,
                                       num_decoder_layers=6,
                                       normalize_before=True,
                                       return_intermediate_dec=False)
        head = DETR(
            num_classes=10,
            num_channels=384,
            transformer=detr_transformer,
            num_queries=40,
            aux_loss=False,
            patch_size=14,
            initial_height=518,
            initial_width=1666,
        )
        full_model = SingleTaskModel(
            backbone=backbone_vit, head=head, freeze_backbone=True)
        full_model.load_state_dict(torch.load(
            "weights/full_model_weights.pth", map_location=self.device))
        if distributed:
            full_model = DDP(full_model, device_ids=[
                self.device], find_unused_parameters=True)

        self.optimizer = AdamW(full_model.parameters(),
                               lr=1e-4, weight_decay=1e-4)
        evaluator = Evaluator(self.dataloader_val,
                              loss=self.loss, callbacks=self.callbacks)
        self.trainer = Trainer(dataset=self.dataloader_train, model=full_model,
                               optimizer=self.optimizer, loss=self.loss, evaluator=evaluator, callbacks=self.callbacks)

    def run(self):
        self.trainer.train()


if __name__ == "__main__":
    full_trainer = TrainRunnerVit(
        distributed=True, output_dir="<path_to_experiment>")
    full_trainer()
