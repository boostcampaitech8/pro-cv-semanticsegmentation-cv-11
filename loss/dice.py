import torch
import torch.nn as nn

class MyDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(MyDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice