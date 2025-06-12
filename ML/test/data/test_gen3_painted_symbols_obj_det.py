# pylint: disable=W0621,R0801
import os

import pytest
import torch
from torch import distributed as dist
from torchvision.transforms import v2  # type: ignore[import-untyped]

from data import CollateFn, Dataset, setup_data_loader
from data.augmentation import NormalizeLabels, TransformImage
from utils import DataKey, LabelType


def setup_dummy_dataset(stage, subset_dim, image_dim):
    return Dataset(
        annotation_file='toy.csv',
        labels_file='toy.json',
        base_dir='smth/',
        transforms=v2.Compose([
            TransformImage(mean=[0.229, 0.224, 0.225], std=[0.485, 0.456, 0.406]),
            NormalizeLabels(*image_dim)
        ]),
        data_type=stage,
        subset_dim=subset_dim,
    )


@pytest.fixture(scope="module")
def image_dim() -> torch.Size:
    return torch.Size([3, 512, 1664])


@pytest.fixture(scope="module")
def class_number() -> int:
    return 13


def initialize_for_distributed():
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl")


def destroy_for_distributed():
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "dataloader_type, dataset_size",
    [
        ("train", 2),
        ("val", 1),
    ],
)
def test_dataset(
        dataloader_type: str,
        image_dim: torch.Size,
        dataset_size: int
) -> None:
    dataset = setup_dummy_dataset(dataloader_type, dataset_size, image_dim[1:])
    assert len(dataset) == dataset_size, "The dataset should be this big"
    for sample in dataset:
        input_img, label, metadata = sample[DataKey.IMAGE], sample[DataKey.LABEL], sample[DataKey.METADATA]
        assert isinstance(input_img, torch.Tensor)
        assert input_img.shape == image_dim, "the image in the correct format"

        assert LabelType.BBOX in label

        label = label[LabelType.BBOX]

        assert isinstance(label, dict)
        assert 'boxes' in label
        assert 'labels' in label
        assert isinstance(metadata, dict)

        assert len(label['boxes']) > 0
        assert len(label['labels']) > 0


@pytest.mark.parametrize(
    "dataloader_type, dataset_size, batch_size, distributed",
    [
        ("train", 2, 2, False),
        ("val", 1, 1, False),
        ("train", 2, 2, True),
        ("val", 1, 1, True),
    ],
)
def test_dataloader(
        dataloader_type: str,
        image_dim: torch.Size, dataset_size: int,
        batch_size: int, *, device: torch.device,
        class_number: int, distributed: bool
) -> None:
    if distributed:
        initialize_for_distributed()
    dataset = setup_dummy_dataset(dataloader_type, dataset_size, image_dim[1:])
    dataloader = setup_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        distributed=distributed,
        collate_fn=CollateFn(device),
        shuffle=False,
        drop_last=False
    )

    for batch, data in enumerate(dataloader):
        x_data = data[DataKey.IMAGE]
        y_data = data[DataKey.LABEL]
        assert isinstance(
            x_data, torch.Tensor), "The images should be a Tensor"
        assert x_data.shape[0] == batch_size, "the number of images should be equal to the batch_size"
        assert x_data.shape[1:] == image_dim, "the shape of the images should be the same"
        assert x_data.device == device, "The device should be correct"
        assert batch < dataset_size // batch_size, "The number of batches should not be greater than the calculated one"
        assert isinstance(y_data, list)
        assert len(
            y_data) == batch_size, "The number of labels should be equal to the batch_size"
        for label in y_data:
            assert isinstance(label, dict), "The labels should be a dict"
            assert "labels" in label, "The labels should be present - the classes"
            assert "boxes" in label, "the boxes should be present"
            classes = label["labels"]
            boxes = label["boxes"]
            assert isinstance(
                classes, torch.Tensor), "The classes should be a Tensor"
            assert isinstance(
                boxes, torch.Tensor), "The coords should be a Tensor"
            assert classes.device == device, "The classes should be on the correct device"
            assert boxes.device == device, "The boxes should be on the correct device"
            if len(classes) > 0:
                assert classes.max().item(
                ) < class_number, "The classes indexes should be smaller than the number of them"
                assert classes.min().item() >= 0, "The classes indexes should be greater or equal to 0"
                assert boxes.max().item() <= 1, "The coords should be normalized -> smaller than 1"
                assert boxes.min().item() >= 0, "The coords should be normalized -> greater than 0"
    if distributed:
        destroy_for_distributed()


def test_wrong_stage() -> None:
    try:
        _ = setup_dummy_dataset("wrong_type", 1, (10, 10))
        assert False
    except ValueError as ve:
        assert str(ve) == "Dataset type is not one of train, val, test!"
