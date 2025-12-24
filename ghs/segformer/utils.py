# utils.py
import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FocalTverskyLoss(nn.Module):
    """
    노트북 그대로: sigmoid 기반 (multilabel 관점)
    targets: (B,C,H,W) one-hot (0/1)
    """
    def __init__(self, alpha=0.6, beta=0.4, gamma=1.1, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(self.eps, 1 - self.eps)
        dims = (0, 2, 3)

        tp = (probs * targets).sum(dim=dims)
        fn = ((1 - probs) * targets).sum(dim=dims)
        fp = (probs * (1 - targets)).sum(dim=dims)

        tversky = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        loss = (1 - tversky) ** self.gamma
        return loss.mean()


class CEFocalTverskyLoss(nn.Module):
    """
    실행 확인 후 추가하려는 CE+FT 조합.
    CE는 multiclass용이라 targets를 argmax로 변환해서 사용.
    """
    def __init__(self, alpha=0.6, beta=0.4, gamma=1.1, ce_weight=0.3, ft_weight=0.7, eps=1e-7, ignore_index=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.ft_weight = ft_weight
        self.ft = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, eps=eps)
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index) if ignore_index is not None else nn.CrossEntropyLoss()

    def forward(self, logits, targets_onehot):
        # targets_onehot: (B,C,H,W)
        targets_ce = targets_onehot.argmax(dim=1).long()  # (B,H,W)
        ce_loss = self.ce(logits, targets_ce)
        ft_loss = self.ft(logits, targets_onehot)
        return self.ce_weight * ce_loss + self.ft_weight * ft_loss


def build_loss(cfg):
    name = cfg["loss"]["name"].lower()
    lcfg = cfg["loss"]

    if name in ["ftloss", "focalltverskyloss", "focal_tversky"]:
        return FocalTverskyLoss(
            alpha=float(lcfg.get("alpha", 0.6)),
            beta=float(lcfg.get("beta", 0.4)),
            gamma=float(lcfg.get("gamma", 1.1)),
        )
    if name in ["ce_ftloss", "ce+ftloss", "ceftloss"]:
        return CEFocalTverskyLoss(
            alpha=float(lcfg.get("alpha", 0.6)),
            beta=float(lcfg.get("beta", 0.4)),
            gamma=float(lcfg.get("gamma", 1.1)),
            ce_weight=float(lcfg.get("ce_weight", 0.3)),
            ft_weight=float(lcfg.get("ft_weight", 0.7)),
            ignore_index=lcfg.get("ignore_index", None),
        )

    raise ValueError(f"Unknown loss: {name}")


@torch.no_grad()
def multilabel_dice_per_class(probs, targets, thr=0.5, eps=1e-6):
    """
    probs   : (B,C,H,W) sigmoid probs
    targets : (B,C,H,W) 0/1
    return  : (C,) dice per class (image-wise averaged)
    """
    preds = (probs > thr).float()

    B, C = preds.shape[:2]
    dices = []
    for c in range(C):
        p = preds[:, c]
        t = targets[:, c]
        inter = (p * t).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dice = (2 * inter + eps) / (union + eps)
        dices.append(dice.mean())
    return torch.stack(dices)  # (C,)


def save_checkpoint(state_dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)


def init_wandb(cfg):
    """
    train.py에서 import wandb 후 사용.
    여기서는 cfg만 정리.
    """
    wcfg = cfg.get("wandb", {})
    use = bool(wcfg.get("use", False))
    return use, wcfg

def mask_to_rle(mask):
    """
    mask: (H, W), values {0,1}
    return: RLE string
    """
    pixels = mask.flatten(order="F")  # column-major
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)