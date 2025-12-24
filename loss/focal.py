import torch
import torch.nn as nn
import torch.nn.functional as F

class MyFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(MyFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        return focal_loss.mean()