from models import SingleTaskModel, DETR, BackboneViT, Transformer
import torch


def build_inference_model(weights: str, device: torch.device) -> SingleTaskModel:
    detr_transformer = Transformer(d_model=384,
                                   dropout=0.2,
                                   nhead=6,
                                   dim_feedforward=2048,
                                   num_decoder_layers=6,
                                   normalize_before=True,
                                   return_intermediate_dec=False)
    head = DETR(
        num_classes=10,
        num_channels=384,
        transformer=detr_transformer,
        num_queries=40,
        aux_loss=False,
        patch_size=14,
        initial_height=518,
        initial_width=1666,
    )
    backbone_vit = BackboneViT(
        embed_dim=384,
        depth=12,
        num_heads=6,
        patch_tokens=True,
        patch_size=14,
        init_values=1e-5,
        img_size=(518, 1666)

    )
    model = SingleTaskModel(backbone_vit, head)
    model.to(device)
    model.load_state_dict(torch.load(
        weights, map_location=device))
    print("Model loaded successfully!")
    model.eval()
    return model
