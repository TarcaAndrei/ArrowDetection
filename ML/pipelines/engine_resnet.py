from data import ArrowDataset, setup_data_loader, CollateFn, ObjectDetectionTransformation
from models import SingleTaskModel, BackboneViT, ResnetBackbone, DETR, Transformer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torchvision.transforms import v2
from loss import HungarianMatcher, DETRLoss
from loops import Trainer, Evaluator
from torch.optim import AdamW
from .base import BaseRunner


class TrainRunner(BaseRunner):
    def __init__(self, distributed):
        super().__init__(distributed)
        distributed = True
        lista_sageti = [
            'left',
            'right',
            'left-right',
        ]
        transforms = ObjectDetectionTransformation()
        dataset_train = ArrowDataset("annotation_file.csv", "path/to/jsons",
                                     lista_sageti, "train",  v2.Compose([transforms]), None)
        dataset_val = ArrowDataset("annotation_file.csv", "path/to/jsons",
                                   lista_sageti, "val",  v2.Compose([transforms]), None)
        collate_fn = CollateFn(self.device)
        dataloader_train = setup_data_loader(
            dataset=dataset_train, batch_size=32, distributed=distributed, collate_fn=collate_fn, shuffle=True, drop_last=True)
        dataloader_val = setup_data_loader(dataset=dataset_val, batch_size=32,
                                           distributed=distributed, collate_fn=collate_fn, shuffle=True, drop_last=True)
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
        cost_class = 2.
        cost_giou = 3.
        cost_bbox = 4.
        number_classes = len(lista_sageti)
        matcher = HungarianMatcher(
            cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
        loss = DETRLoss(num_classes=number_classes, matcher=matcher, eos_coef=0.15,
                        weight_dict=dict(
                            loss_ce=cost_class,
                            loss_bbox=cost_bbox,
                            loss_giou=cost_giou,
                        ),
                        losses=[
                            'labels',
                            'boxes'
                        ],)
        optimizer = AdamW(full_model.parameters(), lr=1e-4, weight_decay=1e-4)
        evaluator = Evaluator(dataloader_val, loss=loss)
        self.trainer = Trainer(dataset=dataloader_train, model=full_model,
                               optimizer=optimizer, loss=loss, evaluator=evaluator)

    def run(self):
        self.trainer.train()


if __name__ == "__main__":
    full_trainer = TrainRunner(distributed=True)
    full_trainer()
