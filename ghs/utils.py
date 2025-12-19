import os
import random
import numpy as np
import torch
import yaml

def load_yaml(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_coef(y_true, y_pred, eps=1e-4):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    return (2. * intersection + eps) / (
        torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps
    )


def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def collect_png_json_pairs(image_root, label_root):
    """
    IMAGE_ROOT, LABEL_ROOT 아래를 재귀적으로 순회하면서
    png / json pair를 수집하고 정렬된 리스트로 반환
    """

    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _, files in os.walk(image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    jsons = {
        os.path.relpath(os.path.join(root, fname), start=label_root)
        for root, _, files in os.walk(label_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }

    # 파일 prefix 기준으로 pair 확인
    png_prefix = {os.path.splitext(f)[0] for f in pngs}
    json_prefix = {os.path.splitext(f)[0] for f in jsons}

    assert len(png_prefix - json_prefix) == 0, "png에 대응되는 json이 없습니다"
    assert len(json_prefix - png_prefix) == 0, "json에 대응되는 png가 없습니다"

    pngs = sorted(pngs)
    jsons = sorted(jsons)

    return pngs, jsons


def collect_test_pngs(image_root):
    """
    test set용 png만 수집 (json 없음)
    """
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _, files in os.walk(image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    return sorted(pngs)

def logits_to_preds(logits, cfg):
    task = cfg["model"].get("task_type", "multilabel")

    if task == "multilabel":
        thr = float(cfg["train"].get("threshold", 0.5))
        probs = torch.sigmoid(logits)
        return (probs > thr).long()

    elif task == "multiclass":
        return logits.argmax(dim=1)

    else:
        raise ValueError(f"Unknown task_type: {task}")

def multiclass_dice_coef(preds, targets, num_classes, eps=1e-5):
    """
    preds:   (B, H, W)           class index
    targets: (B, H, W) OR (B, C, H, W)
    """

    # 🔥 핵심: target이 one-hot이면 class index로 변환
    if targets.dim() == 4:
        # (B, C, H, W) -> (B, H, W)
        targets = targets.argmax(dim=1)

    dice_scores = []

    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

        dice = (2 * intersection + eps) / (union + eps)
        dice_scores.append(dice.mean())

    return torch.stack(dice_scores).mean()

def multiclass_dice_per_class(preds, targets, num_classes, eps=1e-5):
    """
    return: (C,) tensor  — class-wise dice
    """

    if targets.dim() == 4:
        targets = targets.argmax(dim=1)

    dices = []

    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

        dice = (2 * intersection + eps) / (union + eps)
        dices.append(dice.mean())

    return torch.stack(dices)  # (C,)

