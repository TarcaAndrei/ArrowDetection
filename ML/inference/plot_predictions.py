from __future__ import annotations

import torch
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from inference.clase import label_decoder
import cv2
from pathlib import Path
import numpy as np

import collections
import math
import pathlib
import warnings
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont



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

color_skeme = {
    0: 'black',  # turn around
    1: 'crimson',   # left
    2: 'orange',   # left right
    3: 'tomato',   # right
    4: 'brown',  # slight left
    5: 'brown',  # slight right
    6: 'royalblue',  # straight left right
    7: 'mediumblue',  # straight
    8: 'teal',  # straight left
    9: 'darkcyan',  # straight right
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
    clase_interes = list(clase_interes)
    toate_predictiile = []
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
        toate_predictiile = [
            {
                "class": cls_prezis, "bounding_box": bbox_prezis.to(dtype=int).tolist()
            } for cls_prezis, bbox_prezis in zip(box_labels, boxes)
        ]

        final_image = draw_bounding_boxes(
            image, boxes, box_labels, colors=colors_bbox,  font="/teamspace/studios/this_studio/ArrowDetection/ML/inference/OpenSans_Condensed-Medium.ttf", font_size=35, width=4)
            # daca crapa ii de aici :))))))
    else:
        final_image = image.cpu()
    r, g, b = list(final_image)
    if save:
        cv2.imwrite(
            str(Path(output_dir) / f"{name}.jpg"), torch.stack([b, g, r], axis=-1).numpy())
    # else:
    return torch.stack([b, g, r], axis=-1).numpy(), toate_predictiile
    # print("Image plotted successfully! " )


@torch.no_grad()
def draw_bounding_boxes_mine(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
    label_colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
) -> torch.Tensor:

    """
    Draws bounding boxes on given RGB image.
    The image values should be uint8 in [0, 255] or float in [0, 1].
    If fill is True, Resulting Tensor should be saved as PNG image.

    Args:
        image (Tensor): Tensor of shape (C, H, W) and dtype uint8 or float.
        boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
            the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
            `0 <= ymin < ymax < H`.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (color or list of colors, optional): List containing the colors
            of the boxes or single color for all boxes. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for boxes.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.
        label_colors (color or list of colors, optional): Colors for the label text.  See the description of the
            `colors` argument for details.  Defaults to the same colors used for the boxes.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.

    """
    import torchvision.transforms.v2.functional as F  # noqa

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif not (image.dtype == torch.uint8 or image.is_floating_point()):
        raise ValueError(f"The image dtype must be uint8 or float, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")
    elif (boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:, 3]).any():
        raise ValueError(
            "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
        )

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    colors = _parse_colors(colors, num_objects=num_boxes)
    if label_colors:
        label_colors = _parse_colors(label_colors, num_objects=num_boxes)  # type: ignore[assignment]
    else:
        label_colors = colors.copy()  # type: ignore[assignment]

    if font is None:
        if font_size is not None:
            warnings.warn("Argument 'font_size' will be ignored since 'font' is not set.")
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 10)

    # Handle Grayscale images
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    original_dtype = image.dtype
    if original_dtype.is_floating_point:
        image = F.to_dtype(image, dtype=torch.uint8, scale=True)

    img_to_draw = F.to_pil_image(image)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label, label_color in zip(img_boxes, colors, labels, label_colors):  # type: ignore[arg-type]
        if fill:
            fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            margin = width + 1
            draw.text((bbox[0] - 0, bbox[1] - 40), label, fill=label_color, font=txt_font)  # type: ignore[arg-type]

    out = F.pil_to_tensor(img_to_draw)
    if original_dtype.is_floating_point:
        out = F.to_dtype(out, dtype=original_dtype, scale=True)
    return out


