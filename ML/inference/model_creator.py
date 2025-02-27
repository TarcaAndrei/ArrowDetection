from models import SingleTaskModel, DETR, BackboneViT, Transformer
import torch


def build_inference_model(weights: str, device: torch.device, model_type: str = "small") -> SingleTaskModel:
    d_model = 384
    nhead_head = 6
    num_decoder_layer = 6
    backbone_heads = 6
    backbone_embed_dim = 384
    if model_type == "base":
        d_model = 512
        num_decoder_layer = 8
        nhead_head = 8
        backbone_embed_dim = 768
        backbone_heads = 12
    detr_transformer = Transformer(d_model=d_model,
                                   dropout=0.2,
                                   nhead=nhead_head,
                                   dim_feedforward=2048,
                                   num_decoder_layers=num_decoder_layer,
                                   normalize_before=True,
                                   return_intermediate_dec=False)
    head = DETR(
        num_classes=10,
        num_channels=backbone_embed_dim,
        transformer=detr_transformer,
        num_queries=40,
        aux_loss=False,
        patch_size=14,
        initial_height=518,
        initial_width=1666,
    )
    backbone_vit = BackboneViT(
        embed_dim=backbone_embed_dim,
        depth=12,
        num_heads=backbone_heads,
        patch_tokens=True,
        patch_size=14,
        init_values=1e-5,
        img_size=(518, 1666)

    )
    model = SingleTaskModel(backbone_vit, head, freeze_backbone=True)
    model.to(device)
    model.load_state_dict(torch.load(
        weights, map_location=device))
    print("Model loaded successfully!")
    model.eval()
    return model
