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