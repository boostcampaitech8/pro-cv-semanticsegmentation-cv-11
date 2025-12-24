import torch.nn as nn

class MyBCELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MyBCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, predictions, targets):
        return self.loss(predictions, targets)