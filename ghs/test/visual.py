import os
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import build_model
from utils import load_yaml, encode_mask_to_rle, collect_test_pngs

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
    (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
    (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
    (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
    (0, 220, 176),
]

def labelidx2rgb(label_idx, background_idx=0):
    h, w = label_idx.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(PALETTE)):
        if c == background_idx:
            continue
        img[label_idx == c] = PALETTE[c]
    return img

@torch.no_grad()
def visualize_predictions(
    model,
    loader,
    save_dir,
    max_samples=10
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    saved = 0

    for images, image_names in loader:
        images = images.cuda(non_blocking=True)

        logits = model(images)["out"]
        logits = F.interpolate(
            logits, size=(2048, 2048), mode="bilinear", align_corners=False
        )

        preds = logits.argmax(dim=1).cpu().numpy()  # (B, H, W)

        for img, pred, name in zip(images.cpu(), preds, image_names):
            if saved >= max_samples:
                return

            # 원본 이미지 복원
            img_np = img.numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)

            # prediction color mask
            pred_rgb = labelidx2rgb(pred)
            pred_rgb = cv2.resize(
                pred_rgb,
                (img_np.shape[1], img_np.shape[0]),  # (W, H)
                interpolation=cv2.INTER_NEAREST
            )

            # overlay (선택)
            overlay = (0.6 * img_np + 0.4 * (pred_rgb / 255.0))
            overlay = np.clip(overlay, 0, 1)

            pred_rgb_bgr = pred_rgb[:, :, ::-1]
            overlay_bgr = (overlay * 255).astype(np.uint8)[:, :, ::-1]
            img_bgr = (img_np * 255).astype(np.uint8)[:, :, ::-1]

            # side-by-side visualization
            combined = np.concatenate([img_np, overlay], axis=1)  # (H, 2W, 3)

            combined_bgr = (combined * 255).astype(np.uint8)[:, :, ::-1]


            # 저장
            base = os.path.basename(name).replace(".png", "")

            cv2.imwrite(
                os.path.join(save_dir, f"{base}_input_overlay.png"),
                combined_bgr
            )

            cv2.imwrite(
                os.path.join(save_dir, f"{base}_pred.png"),
                pred_rgb_bgr
            )
            saved += 1
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()

class XRayInferenceDataset(Dataset):
    def __init__(self, image_root, pngs, transforms=None):
        self.image_root = image_root
        self.pngs = sorted(pngs)
        self.transforms = transforms

    def __len__(self):
        return len(self.pngs)

    def __getitem__(self, idx):
        rel_path = self.pngs[idx]
        image_path = os.path.join(self.image_root, rel_path)

        image = cv2.imread(image_path)
        image = image / 255.0

        if self.transforms:
            image = self.transforms(image=image)["image"]

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        return image, rel_path


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    IMAGE_ROOT = cfg["data"]["test_image_root"]
    IMAGE_SIZE = int(cfg["data"]["image_size"])
    CKPT_PATH = os.path.join(
        cfg["checkpoint"]["save_dir"],
        cfg["checkpoint"]["filename"]
    )

    tf = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE)])

    pngs = collect_test_pngs(IMAGE_ROOT)

    test_ds = XRayInferenceDataset(
        image_root=IMAGE_ROOT,
        pngs=pngs,
        transforms=tf
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    visualize_predictions(
        model,
        test_loader,
        save_dir="vis_test_preds",
        max_samples=10
    )

if __name__ == "__main__":
    main()
