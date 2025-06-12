from __future__ import annotations

import pytest
import torch

from models import DETR, BackboneViT, LinearHeadClassification, SingleTaskModel, Transformer


def build_detr_dummy_model(image_height, image_width, number_of_classes):
    return SingleTaskModel(
        backbone=BackboneViT(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            patch_tokens=True,
            img_size=(image_height, image_width)
        ),
        head=DETR(
            num_channels=768,
            transformer=Transformer(
                d_model=128,
                dropout=0.2,
                nhead=8,
                dim_feedforward=2048,
                num_decoder_layers=6,
                normalize_before=True,
                return_intermediate_dec=True,
            ),
            num_classes=number_of_classes,
            num_queries=10,
            aux_loss=True,
            initial_height=image_height,
            initial_width=image_width,
            patch_size=16,
        )
    )

@pytest.mark.parametrize(
    'model_builder, image_height, image_width, number_of_classes',
    [
        (build_detr_dummy_model, 512, 1664, 10),
    ]
)
def test_inference(
    model_builder,
    image_height,
    image_width,
    number_of_classes,
    device: torch.device,
):
    model = model_builder(image_height, image_width, number_of_classes).to(device)
    input_data = torch.rand(1, 3, image_height, image_width).to(device)
    model_output = model(input_data)
    assert isinstance(model_output, dict)
    assert 'pred_logits' in model_output
    assert 'pred_boxes' in model_output
    logits = model_output['pred_logits']
    boxes = model_output['pred_boxes']
    assert logits.shape == torch.Size([1, 10, number_of_classes + 1])
    assert boxes.shape == torch.Size([1, 10, 4])
    assert 0 <= boxes.all() <= 1, "These should be normalized"
