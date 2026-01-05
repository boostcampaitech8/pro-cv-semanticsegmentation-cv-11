import torch
import torch.nn as nn
from .class_weights_utils import parse_class_weights, apply_class_weights

class MyJaccardLoss(nn.Module):
    """
    jaccard_loss (IoU Loss)
    """
    def __init__(self, smooth=1e-5, class_weights=None):
        """
        Args:
            smooth: Smoothing factor
            class_weights: Tensor or list of class weights (C,) or dict mapping class names to weights
        """
        super(MyJaccardLoss, self).__init__()
        self.smooth = smooth
        self.class_weights = parse_class_weights(class_weights)

    def forward(self, predictions, targets):
        """
        predictions: (N, C, H, W)
        targets: (N, C, H, W)
        """
        predictions = torch.sigmoid(predictions)
        
        # 클래스별 Jaccard 계산
        B, C, H, W = predictions.shape
        predictions_flat = predictions.view(B, C, -1)  # (N, C, H*W)
        targets_flat = targets.view(B, C, -1)  # (N, C, H*W)
        
        intersection = torch.sum(predictions_flat * targets_flat, dim=2)  # (N, C)
        union = torch.sum(predictions_flat, dim=2) + torch.sum(targets_flat, dim=2) - intersection  # (N, C)
        
        jaccard = (intersection + self.smooth) / (union + self.smooth)  # (N, C)
        jaccard_loss = 1 - jaccard  # (N, C)
        
        # class weights 적용
        if self.class_weights is not None:
            if self.class_weights.device != jaccard_loss.device:
                self.class_weights = self.class_weights.to(jaccard_loss.device)
            jaccard_loss = jaccard_loss * self.class_weights.unsqueeze(0)
        
        # 배치와 클래스에 대해 평균
        return jaccard_loss.mean()