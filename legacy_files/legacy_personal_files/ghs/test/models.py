import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp

# models.py
import torch.nn as nn
from torchvision.models.segmentation import (
    fcn_resnet50,
    FCN_ResNet50_Weights,
)


# class SegFormerWrapper(nn.Module):
#     def __init__(self, pretrained_model_name, num_classes):
#         super().__init__()
#         self.model = SegformerForSemanticSegmentation.from_pretrained(
#             pretrained_model_name,
#             num_labels=num_classes,
#             ignore_mismatched_sizes=True,
#         )

#     def forward(self, x):
#         out = self.model(pixel_values=x).logits
#         # (B, C, h, w) → 원본 사이즈로 업샘플
#         out = nn.functional.interpolate(
#             out,
#             size=x.shape[-2:],
#             mode="bilinear",
#             align_corners=False,
#         )
#         return {"out": out}
class SMPSegFormerWrapper(nn.Module):
    def __init__(self, encoder_name, num_classes, in_channels=3):
        super().__init__()
        self.model = smp.Segformer(
            encoder_name=encoder_name,        # e.g. "mit_b2"
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x):
        out = self.model(x)   # (B, C, H, W)
        return {"out": out}


def build_model(cfg):
    name = cfg["model"]["name"].lower()
    num_classes = cfg["model"]["num_classes"]

    if name == "fcn_resnet50":
        weights = FCN_ResNet50_Weights.DEFAULT
        model = fcn_resnet50(weights=weights)
        model.classifier[4] = nn.Conv2d(512, num_classes, 1)
        return model

    elif name == "segformer":
        return SMPSegFormerWrapper(
            encoder_name=cfg["model"].get("encoder_name", "mit_b2"),
            num_classes=num_classes,
        )

    else:
        raise ValueError(f"Unsupported model: {name}")

