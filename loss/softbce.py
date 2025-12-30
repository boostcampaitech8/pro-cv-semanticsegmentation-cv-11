import torch.nn as nn
import segmentation_models_pytorch as smp

class MySoftBCELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MySoftBCELoss, self).__init__()
        self.loss = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, predictions, targets):
        return self.loss(predictions, targets)