from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class DETRLoss(nn.Module):
    """
    This class computes the loss for DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes: int, matcher: nn.Module,
                 weight_dict: dict[str, float], eos_coef: float, losses: list[str],
                 weight_ce_loss: list | None = None):
        """ 
        Create the criterion.

        Args:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            weight_ce_loss: list with weights for each class, including the no_class
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        if weight_ce_loss is not None:
            empty_weight = torch.tensor(weight_ce_loss)
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]],
                    indices: list[tuple[torch.Tensor, torch.Tensor]], num_boxes: int, log: bool = False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        # initialize default targets as background
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        idx = DETRLoss.get_src_permutation_idx(indices)
        if len(idx[0]) > 0:  # non empty batch
            target_classes[idx] = torch.cat([
                t["labels"][J]
                for t, (_, J) in zip(targets, indices) if len(J) > 0
            ])

        loss_ce = F.cross_entropy(src_logits.transpose(
            1, 2), target_classes, self.empty_weight.to(src_logits.device))
        losses = {'loss_ce': loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]],
                         indices: list[tuple[torch.Tensor, torch.Tensor]], num_boxes: int):  # pylint: disable=all
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]],
                   indices: list[tuple[torch.Tensor, torch.Tensor]], num_boxes: int):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = DETRLoss.get_src_permutation_idx(indices)
        if len(idx[0]) == 0:  # empty batch so set losses to 0
            return {
                # this is needed because DDP complains if there are unused parameters
                'loss_bbox': outputs['pred_boxes'].sum() * 0.,
                'loss_giou': outputs['pred_boxes'].sum() * 0.,
            }

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][i]
            for t, (_, i) in zip(targets, indices) if len(i) > 0
        ], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    @staticmethod
    def get_src_permutation_idx(indices: list[tuple[torch.Tensor, torch.Tensor]]):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss: str,
                 outputs: dict[str, torch.Tensor],
                 targets: list[dict[str, torch.Tensor]],
                 indices: list[tuple[torch.Tensor, torch.Tensor]],
                 num_boxes: int, **kwargs):
        loss_map: dict[str, Callable] = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes_temp = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1  # get_world_size()
        num_boxes = int(torch.clamp(num_boxes_temp / world_size, min=1).item())

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
