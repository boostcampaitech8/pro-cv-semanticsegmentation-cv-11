import torch
import torch.nn as nn
import torch.nn.functional as F
from .class_weights_utils import parse_class_weights, apply_class_weights

class MyBCELoss(nn.Module):
    def __init__(self, class_weights=None, **kwargs):
        """
        Args:
            class_weights: Tensor or list of class weights (C,) or dict mapping class names to weights
                           If dict, will be converted to tensor based on CLASSES order
            **kwargs: Additional arguments for BCEWithLogitsLoss (e.g., pos_weight, reduction)
        """
        super(MyBCELoss, self).__init__()
        
        # reduction='none'으로 설정하여 클래스별 loss를 얻을 수 있도록 함
        reduction = kwargs.pop('reduction', 'none')
        self.loss = nn.BCEWithLogitsLoss(reduction='none', **kwargs)
        
        # class_weights 처리 (공통 유틸리티 사용)
        self.class_weights = parse_class_weights(class_weights)
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        predictions: (N, C, H, W)
        targets: (N, C, H, W)
        """
        # BCE loss 계산 (reduction='none'이므로 (N, C, H, W) 형태)
        loss = self.loss(predictions, targets)  # (N, C, H, W)
        
        # 클래스별로 평균 (spatial dimension에 대해)
        loss_per_class = loss.mean(dim=(0, 2, 3))  # (C,)
        
        # class weights 적용 (공통 유틸리티 사용)
        loss_per_class = apply_class_weights(loss_per_class, self.class_weights)
        
        # 최종 reduction
        if self.reduction == 'mean':
            return loss_per_class.mean()
        elif self.reduction == 'sum':
            return loss_per_class.sum()
        else:
            return loss_per_class.mean()  # default는 mean