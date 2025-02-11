import os
from abc import ABC, abstractmethod

import torch
from torch import distributed as dist
from data import ArrowDataset, setup_data_loader, CollateFn, ObjectDetectionTransformation
import torch
from torchvision.transforms import v2
from loss import HungarianMatcher, DETRLoss
from callbacks import TorchScheduler, LossLogger, ObjectDetectionMAPLogger, EpochLogger, ModelSaver, ObjectDetectionImagePlotter, CallbackCollection
from metrics import ObjectDetectionMAPWrapper


class BaseRunner(ABC):
    def __init__(self, distributed, output_dir):
        self.distributed = distributed
        if distributed:
            dist.init_process_group(backend="nccl")
            self.device = torch.device(int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device=self.device)
            dist.barrier()
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.output_dir = output_dir
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
        self.dataloader_train = setup_data_loader(
            dataset=dataset_train, batch_size=32, distributed=distributed, collate_fn=collate_fn, shuffle=True, drop_last=True)
        self.dataloader_val = setup_data_loader(dataset=dataset_val, batch_size=32,
                                        distributed=distributed, collate_fn=collate_fn, shuffle=True, drop_last=True)
        cost_class = 2.
        cost_giou = 3.
        cost_bbox = 4.
        number_classes = len(lista_sageti)
        matcher = HungarianMatcher(
            cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
        self.loss = DETRLoss(num_classes=number_classes, matcher=matcher, eos_coef=0.15,
                        weight_dict=dict(
                            loss_ce=cost_class,
                            loss_bbox=cost_bbox,
                            loss_giou=cost_giou,
                        ),
                        losses=[
                            'labels',
                            'boxes'
                        ],)
        metrics = ObjectDetectionMAPWrapper({k:v for (k, v) in enumerate(lista_sageti)})
        self.callbacks = CallbackCollection([
            LossLogger(self.output_dir),
            TorchScheduler(gamma=0.98, scheduler_class='ExponentialLR'),
            ObjectDetectionMAPLogger(metrics, self.output_dir),
            EpochLogger(),
            ModelSaver(output_dir=self.output_dir, save_steps=5,),
            ObjectDetectionImagePlotter(label_decoder={k:v for (k, v) in enumerate(lista_sageti)}, output_dir=self.output_dir)
        ])


    @abstractmethod
    def run(self):
        pass

    def end(self):
        if self.distributed:
            dist.destroy_process_group()

    def __call__(self):
        self.run()
        self.end()
