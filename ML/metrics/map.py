from __future__ import annotations

import logging

import torch
from torchmetrics.detection import MeanAveragePrecision

logger = logging.getLogger(__name__)

class ObjectDetectionMAPWrapper:
    def __init__(self, label_decoder: dict[int, str]) -> None:
        """
        Wrapper that computes the mean average precision for object detection task
        Args:
            label_decoder: dict mapping class ids to their original names
        """
        super().__init__()
        self.map = MeanAveragePrecision(iou_type="bbox", box_format="cxcywh", class_metrics=True)
        self.label_decoder = label_decoder

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        image_size: tuple[int, int]
    ) -> None:
        """ 
        This performs the loss computation.
        Args:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted point coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target point coordinates
            image_size: Tuple containing the size of the image

        """
        final_preds = []
        final_targets = []

        device = outputs['pred_logits'].device
        height, width = image_size
        image_sizes = torch.tensor([width, height, width, height], device=device)

        background_index = outputs['pred_logits'].shape[-1] - 1
        for logits, boxes in zip(outputs['pred_logits'], outputs['pred_boxes']):
            scores, indexes = logits.max(-1)
            not_background_objects = indexes != background_index
            final_preds.append({
                "labels": indexes[not_background_objects].int(),
                "boxes": (boxes[not_background_objects] * image_sizes).float(),
                "scores": scores[not_background_objects].float()
            })

        for target in targets:
            labels, boxes = target["labels"], target['boxes']
            if len(boxes) > 0:
                boxes = boxes * image_sizes
            final_targets.append({
                "labels": labels.int(),
                "boxes":  boxes.float(),
            })

        self.map.update(preds=final_preds, target=final_targets)

    def compute(self) -> dict[str, torch.Tensor]:
        """
        Computes and returns the map
        Args:
            more:   if set to true, returns the entire dict
                    else, returns just a tensor with a single value representing the map
        Returns:
            dict of tensors with one element for each map or just one tensor
        """
        logger.info("Computing mAP on %i samples.", len(self.map.groundtruth_labels))
        metrics = self.map.compute()
        classes = metrics.pop('classes')
        map_per_class, mar_100_per_class = metrics.pop('map_per_class'), metrics.pop('mar_100_per_class')

        unseen_classes = list(self.label_decoder.values())
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.label_decoder[class_id.item()]
            metrics[f"class/map_{class_name}"] = class_map
            metrics[f"class/mar_100_{class_name}"] = class_mar
            unseen_classes.remove(class_name)

        for class_name in unseen_classes:
            metrics[f"class/map_{class_name}"] = -1.
            metrics[f"class/mar_100_{class_name}"] = -1.

        return metrics

    def reset(self):
        """
        Clear the states added so far and preparing for a new measuring
        """
        self.map.reset()
