from __future__ import annotations

import torch
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
import cv2
from pathlib import Path

label_decoder = {
    0: 'Turn Around',
    1: 'Left',
    2: 'Left Right',
    3: 'Right',
    4: 'Slight Left',
    5: 'Slight Right',
    6: 'Straight Left Right',
    7: 'Straight',
    8: 'Straight Left',
    9: 'Straight Right',
}


def plot_prediction(image, outputs, name, output_dir):
    height = image.shape[1]
    width = image.shape[2]
    image = ((image-image.min()) / (image.max() -
                                    image.min()) * 255.).to(torch.uint8)
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    probs, class_ids = pred_logits.softmax(-1).max(-1)
    # remove background (last id)
    # id of the last class (background)
    mask = class_ids != pred_logits.shape[-1] - 1
    pred_boxes = pred_boxes[mask]
    class_ids = class_ids[mask]
    probs = probs[mask]
    boxes = pred_boxes.clone()
    if len(boxes) != 0:
        boxes[:, ::2] *= width
        boxes[:, 1::2] *= height
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        if probs is None:
            box_labels = [label_decoder[x.item()]
                          for x in class_ids]
        else:
            box_labels = [
                f'{label_decoder[class_id.item()]}: {round(prob.item(), 4)}'
                for class_id, prob in zip(class_ids, probs)
            ]

        final_image = draw_bounding_boxes(
            image, boxes, box_labels, colors="blue")
    else:
        final_image = image.cpu()
    r, g, b = list(final_image)
    cv2.imwrite(
        str(Path(output_dir) / f"{name}"), torch.stack([b, g, r], axis=-1).numpy())
    print("Image plotted successfully!")
