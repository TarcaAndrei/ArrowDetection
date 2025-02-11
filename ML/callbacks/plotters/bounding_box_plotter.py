from __future__ import annotations

import torch
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

from utils import DataKey, StoreKey

from .base_plotter import ImagePlotter


class ObjectDetectionImagePlotter(ImagePlotter):
    """
    Plotting class for object detection task
    """

    def _draw_image_bbox(self, width: int, height: int, image: torch.Tensor, *,
                         boxes: torch.Tensor, class_ids: torch.Tensor, probs: torch.Tensor | None = None
                         ) -> torch.Tensor:
        """
        function to draw a set of boxes on an image
        Args:
            width: the width of the image - for rescaling
            height: the height of the image
            image: the initial image
            boxes: Tensor of shape [nr_instances, K, 2] - representing the keypoints
            class_ids: the indexes of the labels
            probs: the probabilities of the predictions
        Returns:
            a Tensor representing the image with the points
        """
        if len(boxes) == 0:
            return image.cpu()
        boxes[:, ::2] *= width
        boxes[:, 1::2] *= height
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        if probs is None:
            box_labels = [self.label_decoder[x.item()]
                          for x in class_ids]
        else:
            box_labels = [
                f'{self.label_decoder[class_id.item()]}: {round(prob.item(), 4)}'
                for class_id, prob in zip(class_ids, probs)
            ]

        return draw_bounding_boxes(image, boxes, box_labels, colors="red")

    def _draw_images(self, store):
        batch_images = store[StoreKey.DATA][DataKey.IMAGE]
        batch_labels = store[StoreKey.DATA][DataKey.LABEL]
        batch_pred_logits = store[StoreKey.OUTPUT]['pred_logits']
        batch_pred_boxes = store[StoreKey.OUTPUT]['pred_boxes']

        height, width = batch_images.shape[-2:]
        plots = []
        for image, labels, pred_logits, pred_boxes in zip(
            batch_images,
            batch_labels,
            batch_pred_logits,
            batch_pred_boxes
        ):
            image = ((image-image.min()) / (image.max() -
                     image.min()) * 255.).to(torch.uint8)

            gt_image = self._draw_image_bbox(width, height, image.clone(),
                                             boxes=labels['boxes'].clone(), class_ids=labels['labels'])
            probs, class_ids = pred_logits.softmax(-1).max(-1)
            # remove background (last id)
            # id of the last class (background)
            mask = class_ids != pred_logits.shape[-1] - 1
            pred_boxes = pred_boxes[mask]
            class_ids = class_ids[mask]
            probs = probs[mask]
            pred_image = self._draw_image_bbox(width, height, image.clone(),
                                               boxes=pred_boxes.clone(), class_ids=class_ids, probs=probs)

            plots.append(torch.concat([gt_image, pred_image], axis=1))

        return plots
