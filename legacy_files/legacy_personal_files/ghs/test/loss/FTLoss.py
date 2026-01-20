import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4, gamma=1.1, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        # logits, targets: (B,C,H,W)
        probs = torch.sigmoid(logits)
        probs = probs.clamp(self.eps, 1 - self.eps)

        dims = (0, 2, 3)  # 배치+공간합, 클래스별로 남김
        tp = (probs * targets).sum(dim=dims)
        fn = ((1 - probs) * targets).sum(dim=dims)
        fp = (probs * (1 - targets)).sum(dim=dims)

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = (1 - tversky) ** self.gamma

        return loss[1:].mean()