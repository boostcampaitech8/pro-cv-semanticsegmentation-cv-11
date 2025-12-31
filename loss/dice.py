# import torch
# import torch.nn as nn

# class MyDiceLoss(nn.Module):
#     def __init__(self, smooth=1e-5):
#         super(MyDiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, predictions, targets):
#         predictions = torch.sigmoid(predictions)
#         predictions = predictions.contiguous().view(-1)
#         targets = targets.contiguous().view(-1)
        
#         intersection = (predictions * targets).sum()
#         dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
#         return 1 - dice

import torch
import torch.nn as nn

class MyDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits:  (N, 1, H, W)
        targets: (N, 1, H, W)
        """
        # from_logits=True
        probs = torch.sigmoid(logits)

        # batch-wise가 아니라 sample-wise Dice
        dims = (1, 2, 3)  # channel, H, W

        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs + targets, dims)

        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        # smp DiceLoss는 (1 - dice)의 mean
        loss = 1 - dice.mean()

        return loss
