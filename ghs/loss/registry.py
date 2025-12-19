from .bce_dice import BCEDiceLoss
from .softmax_dice import SoftmaxDiceLoss
from .ce_dice import CEDiceLoss
import torch.nn as nn

LOSS_REGISTRY = {
    "bcediceloss": BCEDiceLoss,
    "softmaxdiceloss": SoftmaxDiceLoss,
    "cediceloss": CEDiceLoss,
    "crossentropyloss": nn.CrossEntropyLoss,
}

def build_loss(cfg):
    name = cfg["loss"]["name"].lower()
    params = cfg["loss"].copy()
    params.pop("name")

    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}")

    return LOSS_REGISTRY[name](**params)
