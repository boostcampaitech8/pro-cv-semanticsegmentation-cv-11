# inference.py
import os
import argparse
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

from models import build_model
from dataset import XRayInferenceDataset
from utils import load_yaml

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

IND2CLASS = {i: c for i, c in enumerate(CLASSES)}

def encode_mask_to_rle(mask: np.ndarray) -> str:
    """
    mask: (H, W) uint8 {0,1}
    NOTE: 노트북과 동일하게 column-major(F)로 인코딩
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/custom.yaml")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    size = int(cfg["data"]["image_size"])
    thr = float(cfg["train"].get("threshold", 0.5))

    test_root = cfg["data"]["test_image_root"]

    # ✅ yaml에서 inference 설정 읽기 (없으면 기본값)
    infer_cfg = cfg.get("inference", {})
    ckpt_path = infer_cfg.get(
        "checkpoint",
        os.path.join(cfg["checkpoint"]["save_dir"], cfg["checkpoint"]["filename"])
    )
    out_csv = infer_cfg.get("out_csv", "./output.csv")
    batch_size = int(infer_cfg.get("batch_size", 2))

    tf = A.Compose([
        A.Resize(size, size),
        A.Normalize(),
        ToTensorV2(),
    ])

    # test 이미지 수집 (노트북처럼)
    pngs = []
    for root, _, files in os.walk(test_root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, test_root)
                pngs.append(rel)
    pngs = sorted(pngs)

    ds = XRayInferenceDataset(test_root, pngs, transforms=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(cfg).cuda()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    rles = []
    filename_and_class = []

    with torch.no_grad():
        for images, image_names in tqdm(loader, desc="infer", total=len(loader)):
            images = images.cuda(non_blocking=True)

            logits = model(images)  # (B, C, H, W)  ✅ smp segformer면 이게 그대로 logits
            # ✅ 노트북과 동일: 제출 마스크 해상도는 2048x2048
            logits = F.interpolate(logits, size=(2048, 2048), mode="bilinear", align_corners=False)

            probs = torch.sigmoid(logits)
            preds = (probs > thr).to(torch.uint8).cpu().numpy()  # (B,C,2048,2048) 0/1

            for pred, image_name in zip(preds, image_names):
                # image_name은 파일명만 남기기(노트북은 나중에 basename 처리)
                for c, segm in enumerate(pred):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{os.path.basename(image_name)}")

    # ✅ 노트북과 동일한 CSV 생성
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    df.to_csv(out_csv, index=False)
    print(f"[DONE] saved submission CSV -> {out_csv}")

if __name__ == "__main__":
    main()
