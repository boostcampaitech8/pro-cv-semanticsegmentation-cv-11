# import torch
# import torch.nn as nn

# class MyDiceLoss(nn.Module):
#     def __init__(self, smooth=1e-5):
#         super(MyDiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, predictions, targets):
#         predictions = torch.sigmoid(predictions)
#         predictions = predictions.contiguous().view(-1)
#         targets = targets.contiguous().view(-1)
        
#         intersection = (predictions * targets).sum()
#         dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
#         return 1 - dice

import torch
import torch.nn as nn
from .class_weights_utils import parse_class_weights, apply_class_weights

class MyDiceLoss(nn.Module):
    def __init__(self, smooth=1e-4, class_weights=None):
        """
        Args:
            smooth: Smoothing factor for dice calculation
            class_weights: Tensor or list of class weights (C,) or dict mapping class names to weights
                           If dict, will be converted to tensor based on CLASSES order
        """
        super().__init__()
        self.smooth = smooth
        
        # class_weights 처리 (공통 유틸리티 사용)
        self.class_weights = parse_class_weights(class_weights)

    def forward(self, logits, targets):
        """
        logits:  (N, C, H, W) - C는 클래스 수 (29)
        targets: (N, C, H, W)
        """
        # from_logits=True
        probs = torch.sigmoid(logits)

        # 클래스별 Dice 계산
        # probs: (N, C, H, W), targets: (N, C, H, W)
        B, C, H, W = probs.shape
        
        probs_flat = probs.view(B, C, -1)  # (N, C, H*W)
        targets_flat = targets.view(B, C, -1)  # (N, C, H*W)
        
        intersection = torch.sum(probs_flat * targets_flat, dim=2)  # (N, C)
        cardinality = torch.sum(probs_flat + targets_flat, dim=2)  # (N, C)
        
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)  # (N, C)
        dice_loss = 1 - dice  # (N, C)
        
        # class weights 적용 (공통 유틸리티 사용)
        # (N, C) 형태이므로 배치별로 적용
        if self.class_weights is not None:
            # device 맞추기
            if self.class_weights.device != dice_loss.device:
                self.class_weights = self.class_weights.to(dice_loss.device)
            # (N, C) * (C,) -> (N, C)
            dice_loss = dice_loss * self.class_weights.unsqueeze(0)
        
        # 배치와 클래스에 대해 평균
        return dice_loss.mean()
