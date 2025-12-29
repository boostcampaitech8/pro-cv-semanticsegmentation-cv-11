import torch
import torch.nn as nn

class MyFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-5):
        """
        Focal Tversky Loss
        
        Args:
            alpha: False Positive에 대한 가중치 (기본값: 0.3)
            beta: False Negative에 대한 가중치 (기본값: 0.7)
            gamma: Focal parameter, 어려운 샘플에 집중하는 정도 (기본값: 2.0)
            smooth: 수치 안정성을 위한 smoothing factor (기본값: 1e-5)
        """
        super(MyFocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) - 모델 출력 (logits)
            targets: (B, C, H, W) - Ground truth
        
        Returns:
            focal_tversky_loss: scalar
        """
        predictions = torch.sigmoid(predictions)
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        # Tversky coefficient 계산
        TP = (predictions * targets).sum()
        FP = ((1 - targets) * predictions).sum()
        FN = (targets * (1 - predictions)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Focal Tversky Loss: (1 - Tversky)^gamma
        focal_tversky_loss = (1 - tversky) ** self.gamma
        
        return focal_tversky_loss

