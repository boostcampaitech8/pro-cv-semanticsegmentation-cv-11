import torch
import torch.nn as nn
import torch.nn.functional as F
from .class_weights_utils import parse_class_weights, apply_class_weights

class MyFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        """
        Args:
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            class_weights: Tensor or list of class weights (C,) or dict mapping class names to weights
        """
        super(MyFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = parse_class_weights(class_weights)

    def forward(self, predictions, targets):
        """
        predictions: (N, C, H, W)
        targets: (N, C, H, W)
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )  # (N, C, H, W)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss  # (N, C, H, W)
        
        # 클래스별로 평균 (spatial dimension에 대해)
        loss_per_class = focal_loss.mean(dim=(0, 2, 3))  # (C,)
        
        # class weights 적용
        loss_per_class = apply_class_weights(loss_per_class, self.class_weights)
        
        return loss_per_class.mean()