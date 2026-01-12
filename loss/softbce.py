import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .class_weights_utils import parse_class_weights, apply_class_weights

class MySoftBCELoss(nn.Module):
    def __init__(self, class_weights=None, **kwargs):
        """
        Args:
            class_weights: Tensor or list of class weights (C,) or dict mapping class names to weights
            **kwargs: Additional arguments for SoftBCEWithLogitsLoss
        """
        super(MySoftBCELoss, self).__init__()
        self.loss = smp.losses.SoftBCEWithLogitsLoss(**kwargs)
        self.class_weights = parse_class_weights(class_weights)

    def forward(self, predictions, targets):
        """
        predictions: (N, C, H, W)
        targets: (N, C, H, W)
        """
        # SoftBCE loss 계산
        loss = self.loss(predictions, targets)  # (N, C, H, W) 또는 (N, C, H, W)
        
        # 클래스별로 평균 (spatial dimension에 대해)
        if loss.dim() == 4:
            loss_per_class = loss.mean(dim=(0, 2, 3))  # (C,)
        else:
            # 이미 평균된 경우
            loss_per_class = loss
        
        # class weights 적용
        loss_per_class = apply_class_weights(loss_per_class, self.class_weights)
        
        return loss_per_class.mean()