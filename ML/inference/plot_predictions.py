from __future__ import annotations

import torch
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from inference.clase import label_decoder
import cv2
from pathlib import Path
import numpy as np

color_skeme = {
    0: 'black',
    1: 'red',
    2: 'red',
    3: 'red',
    4: 'brown',
    5: 'brown',
    6: 'blue',
    7: 'blue',
    8: 'blue',
    9: 'blue',
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
            image, boxes, box_labels, colors="red")
    else:
        final_image = image.cpu()
    r, g, b = list(final_image)
    cv2.imwrite(
        str(Path(output_dir) / f"{name}"), torch.stack([b, g, r], axis=-1).numpy())
    # print("Image plotted successfully! " )


def plot_frame_prediction(image, outputs, name=None, output_dir=None, save=True, confidenta=0.0, clase_interes=label_decoder.keys()):
    image = torch.tensor(image)
    image = image.permute(2, 0, 1)
    height = image.shape[1]
    width = image.shape[2]
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    probs, class_ids = pred_logits.softmax(-1).max(-1)
    # remove background (last id)
    # id of the last class (background)
    mask = class_ids != pred_logits.shape[-1] - 1
    pred_boxes = pred_boxes[mask]
    class_ids = class_ids[mask]
    probs = probs[mask]

    # confidenta modelului
    mask_confidenta = probs >= confidenta
    pred_boxes = pred_boxes[mask_confidenta]
    class_ids = class_ids[mask_confidenta]
    probs = probs[mask_confidenta]

    # clase interes
    mask_clase = np.isin(class_ids.cpu(), clase_interes)
    pred_boxes = pred_boxes[mask_clase]
    class_ids = class_ids[mask_clase]
    probs = probs[mask_clase]

    boxes = pred_boxes.clone()
    if len(boxes) != 0:
        boxes[:, ::2] *= width
        boxes[:, 1::2] *= height
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        if probs is None:
            box_labels = [label_decoder[x.item()]
                          for x in class_ids]
            colors_bbox = [color_skeme[x.item()]
                           for x in class_ids
                           ]
        else:
            box_labels = [
                f'{label_decoder[class_id.item()]}'
                for class_id, prob in zip(class_ids, probs)
            ]
            colors_bbox = [
                color_skeme[class_id.item()] for class_id in class_ids
            ]

        final_image = draw_bounding_boxes(
            image, boxes, box_labels, colors=colors_bbox)
    else:
        final_image = image.cpu()
    r, g, b = list(final_image)
    if save:
        cv2.imwrite(
            str(Path(output_dir) / f"{name}.jpg"), torch.stack([b, g, r], axis=-1).numpy())
    else:
        return torch.stack([b, g, r], axis=-1).numpy()
    # print("Image plotted successfully! " )
