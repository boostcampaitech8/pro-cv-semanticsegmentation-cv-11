import torch
import torch.nn as nn
from .class_weights_utils import parse_class_weights, apply_class_weights

class MyFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-5, class_weights=None):
        """
        Focal Tversky Loss
        
        Args:
            alpha: False Positive에 대한 가중치 (기본값: 0.3)
            beta: False Negative에 대한 가중치 (기본값: 0.7)
            gamma: Focal parameter, 어려운 샘플에 집중하는 정도 (기본값: 2.0)
            smooth: 수치 안정성을 위한 smoothing factor (기본값: 1e-5)
            class_weights: Tensor or list of class weights (C,) or dict mapping class names to weights
        """
        super(MyFocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.class_weights = parse_class_weights(class_weights)

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) - 모델 출력 (logits)
            targets: (B, C, H, W) - Ground truth
        
        Returns:
            focal_tversky_loss: scalar
        """
        predictions = torch.sigmoid(predictions)
        
        # 클래스별 Focal Tversky 계산
        B, C, H, W = predictions.shape
        predictions_flat = predictions.view(B, C, -1)  # (N, C, H*W)
        targets_flat = targets.view(B, C, -1)  # (N, C, H*W)
        
        TP = torch.sum(predictions_flat * targets_flat, dim=2)  # (N, C)
        FP = torch.sum((1 - targets_flat) * predictions_flat, dim=2)  # (N, C)
        FN = torch.sum(targets_flat * (1 - predictions_flat), dim=2)  # (N, C)
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)  # (N, C)
        focal_tversky_loss = (1 - tversky) ** self.gamma  # (N, C)
        
        # class weights 적용
        if self.class_weights is not None:
            if self.class_weights.device != focal_tversky_loss.device:
                self.class_weights = self.class_weights.to(focal_tversky_loss.device)
            focal_tversky_loss = focal_tversky_loss * self.class_weights.unsqueeze(0)
        
        # 배치와 클래스에 대해 평균
        return focal_tversky_loss.mean()

