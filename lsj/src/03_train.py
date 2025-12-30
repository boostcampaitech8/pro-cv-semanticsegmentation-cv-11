#jupyter command 에서 library download 하기
# !pip install git+https://github.com/qubvel/segmentation_models.pytorch

import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import os, sys
sys.path.append(os.getcwd())  # ✅ 프로젝트 루트를 path에 추가
from src.unet3plus import UNet_3Plus, UNet_3Plus_DeepSup , UNet_3Plus_DeepSup_CGM # 추가

NUM_CLASSES = 29
MODEL_ARCH = "unet3plus_deepsup"      # "unetpp" | "unet3plus" | "unet3plus_deepsup" | "unet3plus_deepsup_cgm"

if MODEL_ARCH == "unetpp":
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )

elif MODEL_ARCH == "unet3plus":
    # ✅ logits 출력 (너의 loss가 BCEWithLogits 계열이므로 이게 맞음)
    model = UNet_3Plus(
        in_channels=3,
        n_classes=NUM_CLASSES,
        final_activation=None,
    )

elif MODEL_ARCH == "unet3plus_deepsup":
    # ⚠️ (d1,d2,d3,d4,d5) 튜플 반환
    model = UNet_3Plus_DeepSup(
        in_channels=3,
        n_classes=NUM_CLASSES,
        final_activation=None,
    )

elif MODEL_ARCH == "unet3plus_deepsup_cgm":
    # ⚠️ (d1,d2,d3,d4,d5) 튜플 반환
    model = UNet_3Plus_DeepSup_CGM(
        in_channels=3,
        n_classes=NUM_CLASSES,
        final_activation=None,
    )

from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
# Loss function을 정의합니다.

dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
bce  = smp.losses.SoftBCEWithLogitsLoss()

class ComboLoss(nn.Module):
    def __init__(self, w_dice=0.7, w_bce=0.3):
        super().__init__()
        self.dice = dice
        self.bce = bce
        self.w_dice = w_dice
        self.w_bce = w_bce
    def forward(self, y_pred, y_true):
        return self.w_dice * self.dice(y_pred, y_true) + self.w_bce * self.bce(y_pred, y_true)

criterion = ComboLoss()

LOSS_DICT = {
    "softbce": smp.losses.SoftBCEWithLogitsLoss(),
    # Tversky here is DiceLoss :)
    "tversky": smp.losses.TverskyLoss(mode="multilabel", log_loss=False),
    "focal": smp.losses.FocalLoss(mode="multilabel"),
    "jaccard": smp.losses.JaccardLoss(mode="multilabel"),
}

class SumLoss(nn.Module):
    def __init__(self, losses: dict, weights: dict | None = None):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {k: 1.0 for k in losses.keys()}

        # weights에 없는 키가 있으면 기본 1.0
        for k in self.losses.keys():
            self.weights.setdefault(k, 1.0)

    def forward(self, y_pred, y_true):
        total = 0.0
        for name, loss_fn in self.losses.items():
            w = self.weights.get(name, 1.0)
            total = total + w * loss_fn(y_pred, y_true)
        return total

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

        tversky = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        loss = (1 - tversky) ** self.gamma

        return loss.mean()

# 전부 동일 가중치로 단순 합
# criterion = SumLoss(LOSS_DICT)
# criterion = FocalTverskyLoss(alpha=0.6, beta=0.4, gamma=1.1)
# criterion = smp.losses.JaccardLoss(
#     mode="multilabel",     # ✅ NUM_CLASSES=29 → multilabel
#     from_logits=True
# )



# Optimizer를 정의합니다.

# import torch_optimizer as optim

# optimizer = optim.AdamP(
#     model.parameters(),
#     lr=LR,
#     weight_decay=1e-6,
#     betas=(0.9, 0.999)
# )

optimizer = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=1e-6)


scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",       # dice를 최대화하니까 max
    factor=0.5,       # lr 줄이는 비율 (예: 절반)
    patience=3,       # 몇 번 개선 없으면 줄일지
    threshold=1e-4,   # 개선 판정 임계
    cooldown=0,
    min_lr=1e-7,
    verbose=True,
)

# scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=800, 
#                                           # 10 epochs worth of steps 
#                                           cycle_mult=1.0, 
#                                           # # Same cycle length 
#                                           max_lr=LR, 
#                                           # # Use current LR (5e-4) as max 
#                                           min_lr=1e-6, 
#                                           # # Minimum LR 
#                                           warmup_steps=1, 
#                                           # # ~5-10% of first cycle for warmup 50 
#                                           gamma=0.5)

# 시드를 설정합니다.
set_seed()

wandb.init(
    entity="cv_11",
    project="cv-11-SEG",
    name=f"lsj_{model.__class__.__name__}_epochs{NUM_EPOCHS}",
    config={
        "optimizer": optimizer.__class__.__name__,
        "loss": criterion.__class__.__name__,
        "lr": LR,
        "epochs": NUM_EPOCHS,
        "model": model.__class__.__name__,
    }
)

train(model, train_loader, valid_loader, criterion, optimizer, scheduler)
