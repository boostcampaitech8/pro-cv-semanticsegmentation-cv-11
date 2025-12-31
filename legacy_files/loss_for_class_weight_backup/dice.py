import torch
import torch.nn as nn

class MyDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, class_weights=None):
        super(MyDiceLoss, self).__init__()
        self.smooth = smooth

        ### 251227 추가: .yaml로부터 class weight를 받을 수 있도록 만듦.###
        # class_weights가 리스트로 들어오면 tensor로 변환
        if class_weights is not None and isinstance(class_weights, list):
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = class_weights
        #################################################################

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # 기존
        # predictions = predictions.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)

        # 각 클래스별로 dice 계산 (multilabel)
        # predictions: (B, C, H, W)
        # targets: (B, C, H, W)
        B, C, H, W = predictions.shape
        
        predictions = predictions.view(B, C, -1)  # (B, C, H*W)
        targets = targets.view(B, C, -1)  # (B, C, H*W)
        
        intersection = (predictions * targets).sum(dim=2)  # (B, C)
        union = predictions.sum(dim=2) + targets.sum(dim=2)  # (B, C)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # (B, C)
        dice_loss = 1 - dice  # (B, C)
        
        # 기존
        # intersection = (predictions * targets).sum()
        # dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        # class weights 적용
        if self.class_weights is not None:
            if self.class_weights.device != dice_loss.device:
                self.class_weights = self.class_weights.to(dice_loss.device)
            dice_loss = dice_loss * self.class_weights.unsqueeze(0)  # (B, C) * (1, C)
        
        # 기존
        # return 1 - dice

        # 배치와 클래스에 대해 평균
        return dice_loss.mean()