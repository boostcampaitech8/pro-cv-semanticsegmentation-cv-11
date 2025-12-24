import torch.nn as nn
from .softmax_dice import SoftmaxDiceLoss

class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight=0.5, ignore_index=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = SoftmaxDiceLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        dice = self.dice(logits, target)
        return self.ce_weight * ce + (1 - self.ce_weight) * dice
