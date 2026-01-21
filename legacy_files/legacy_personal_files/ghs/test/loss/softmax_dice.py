import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        target_oh = F.one_hot(target, num_classes).permute(0,3,1,2).float()

        if self.ignore_index is not None:
            mask = (target != self.ignore_index).unsqueeze(1)
            probs = probs * mask
            target_oh = target_oh * mask

        dims = (0,2,3)
        intersection = (probs * target_oh).sum(dims)
        union = probs.sum(dims) + target_oh.sum(dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
