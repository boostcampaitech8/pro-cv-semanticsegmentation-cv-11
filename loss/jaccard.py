import torch
import torch.nn as nn

class MyJaccardLoss(nn.Module):
    """
    jaccard_loss
    """
    def __init__(self, smooth=1e-5):
        super(MyJaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - jaccard