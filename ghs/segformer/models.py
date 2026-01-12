# models.py
import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(cfg) -> nn.Module:
    mcfg = cfg["model"]

    name = mcfg["name"].lower()
    encoder_name = mcfg.get("encoder_name", "mit_b4")
    encoder_weights = mcfg.get("encoder_weights", "imagenet")
    in_channels = int(mcfg.get("in_channels", 3))
    classes = int(mcfg["num_classes"])

    if name == "segformer":
        model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        return model

    elif name in ["unetplusplus", "unet++", "unetpp"]:
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    else:
        raise ValueError(f"Unsupported model: {name}")
