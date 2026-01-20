from .bce_dice import BCEDiceLoss
from .softmax_dice import SoftmaxDiceLoss
from .ce_dice import CEDiceLoss
from .FTLoss import FocalTverskyLoss
from .ce_ftloss import CEFocalTverskyLoss
import torch.nn as nn

LOSS_REGISTRY = {
    "bcediceloss": BCEDiceLoss,
    "softmaxdiceloss": SoftmaxDiceLoss,
    "cediceloss": CEDiceLoss,
    "crossentropyloss": nn.CrossEntropyLoss,
    "ftloss": FocalTverskyLoss,
    "ce_ftloss": CEFocalTverskyLoss,
}

def build_loss(cfg):
    name = cfg["loss"]["name"].lower()
    params = cfg["loss"].copy()
    params.pop("name")

    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}")

    return LOSS_REGISTRY[name](**params)
