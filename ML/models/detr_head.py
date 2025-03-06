# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .transformer import Transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, num_channels: int, transformer: Transformer,
                 num_classes: int, num_queries: int, *,
                 aux_loss: bool = False, initial_width: int = 1664, initial_height: int = 512,
                 patch_size: int = 16, resnet_backbone: bool = False):
        """ Initializes the model.
        Parameters:
            num_channels: int representing the number of channels from the backbone
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: if set to true, outputs for auxiliary loss will be provided
            initial_width: the width of the image
            initial_height: the height of the image
            patch_size: the patch size used in the backbone transformer
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
        self.num_channels = num_channels
        self.aux_loss = aux_loss
        self.initial_width = initial_width
        self.initial_height = initial_height
        self.patch_size = patch_size
        self.resnet_backbone = resnet_backbone

    def forward(self, backbone_outputs: torch.Tensor, h0: int | None = None, w0: int | None = None) -> dict[str, torch.Tensor]:
        """ 
        Args:
            backbone_outputs: a tensor representing the results from the backbone part
        Returns:
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        hs = self._forward_transformer(backbone_outputs, h0=h0, w0=w0)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)
        return out

    def _forward_transformer(self, backbone_outputs: torch.Tensor, h0: int | None = None, w0: int | None = None) -> torch.Tensor:
        """
        Function that forwards the backbone outputs through the transformer part
        Args:
            backbone_outputs: a tensor representing the results from the backbone part
        Returns:
            The enriched object queries after the transformer part
        """
        if self.resnet_backbone:
            features, pos = backbone_outputs
            src, mask = features[-1].decompose()
            assert mask is not None
            hs = self.transformer(self.input_proj(src), mask,
                                  self.query_embed.weight, pos[-1])[0]
            return hs
        backbone_outputs = backbone_outputs.permute(0, 2, 1).contiguous()
        # we want the tokens to be that
        if h0 is None and w0 is None:
            h0 = self.initial_height // self.patch_size
            w0 = self.initial_width // self.patch_size
        src = backbone_outputs.reshape(-1, self.num_channels, h0, w0)
        pos = None
        b, _, h, w = src.shape
        # we don't padd the values since the backbone_outputs are already a tensor
        projection = self.input_proj(src)
        b, _, h, w = projection.shape
        mask = torch.zeros([b, h, w], dtype=torch.bool, device=src.device)
        assert mask is not None
        hs = self.transformer(projection, mask,
                              self.query_embed.weight, pos)[0]
        return hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
