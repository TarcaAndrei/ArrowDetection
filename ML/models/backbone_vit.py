
from functools import partial

import torch

from .layers import NestedTensorBlock as Block
from .layers.attention import MemEffAttention
from .vision_transformer import DinoVisionTransformer


class BackboneViT(torch.nn.Module):
    """
    Generic Backbone for Vision Transformer models.
    Instantiates a DinoVisionTransformer with configurable options.
    """

    def __init__(self, patch_tokens: bool = False, **kwargs):
        """
        Args:
            patch_tokens: Whether to return only the class token or to also add the patch tokens.
        """
        super().__init__()
        self.patch_tokens = patch_tokens
        self.vit = DinoVisionTransformer(
            block_fn=partial(Block, attn_class=MemEffAttention),
            **kwargs
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward function for backbone.
        
        Args:
            inputs (torch.Tensor): Inputs to the model.

        Returns:
            torch.Tensor: Class token or patch tokens of the model output.
        """
        model_output = self.vit(inputs)

        if self.patch_tokens:
            only_patch_tokens = model_output["x_norm_patchtokens"]
            only_cls_tokens = model_output["x_norm_clstoken"].unsqueeze(1)
            return only_patch_tokens + only_cls_tokens
        return model_output["x_norm_clstoken"]

    def load_state_dict(self, state_dict, *args, **kwargs):
        print("Loading backbone..")
        def _process_weights(new_weights, old_weights):
            if "pos_embed" in new_weights and "pos_embed" in old_weights:
                if new_weights["pos_embed"].shape != old_weights["pos_embed"].shape:
                    new_weights["pos_embed"] = new_weights["pos_embed"].unsqueeze(0)
                    new_weights["pos_embed"] = torch.nn.functional.interpolate(
                        new_weights["pos_embed"],
                        size=(old_weights["pos_embed"].shape[1],
                            old_weights["pos_embed"].shape[2]),
                        mode="bicubic",
                        align_corners=False,
                    )
                    new_weights["pos_embed"] = new_weights["pos_embed"].squeeze(0)
            return new_weights

        preprocessed_state_dict = _process_weights(
            new_weights = state_dict,
            old_weights = self.vit.state_dict()
        )
        self.vit.load_state_dict(preprocessed_state_dict, *args, **kwargs)
