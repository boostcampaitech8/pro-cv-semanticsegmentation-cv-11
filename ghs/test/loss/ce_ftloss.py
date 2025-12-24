import torch.nn as nn
from .FTLoss import FocalTverskyLoss

class CEFocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha=0.6,
        beta=0.4,
        gamma=1.1,
        ce_weight=0.3,
        ft_weight=0.7,
        eps=1e-7
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.ft_weight = ft_weight

        self.ce = nn.CrossEntropyLoss()
        self.ft = FocalTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            eps=eps
        )

    def forward(self, logits, targets):
        """
        logits  : (B, C, H, W)
        targets : (B, C, H, W) one-hot
        """
        # CE는 class index 필요
        targets_ce = targets.argmax(dim=1).long()

        ce_loss = self.ce(logits, targets_ce)
        ft_loss = self.ft(logits, targets)

        return self.ce_weight * ce_loss + self.ft_weight * ft_loss
