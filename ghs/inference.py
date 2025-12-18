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

# ---------------- CLASSES (고정) ----------------
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
# ------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


# ---------------- Dataset ----------------
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


# ---------------- Inference ----------------
@torch.no_grad()
def run_inference(model, loader, thr):
    model.eval()
    rles = []
    filename_and_class = []

    for images, image_names in tqdm(loader):
        images = images.cuda(non_blocking=True)

        logits = model(images)["out"]
        logits = F.interpolate(logits, size=(2048, 2048), mode="bilinear")

        probs = torch.sigmoid(logits).cpu().numpy()

        for prob, img_name in zip(probs, image_names):
            for cls_idx, cls_mask in enumerate(prob):
                rle = encode_mask_to_rle(cls_mask > thr)
                rles.append(rle)
                filename_and_class.append(f"{CLASSES[cls_idx]}_{img_name}")

    return rles, filename_and_class


# ---------------- Main ----------------
def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    IMAGE_ROOT = cfg["data"]["test_image_root"]
    IMAGE_SIZE = int(cfg["data"]["image_size"])
    THR = float(cfg["train"].get("threshold", 0.5))

    CKPT_PATH = os.path.join(
        cfg["checkpoint"]["save_dir"],
        cfg["checkpoint"]["filename"]
    )

    OUTPUT_DIR = "outputs"
    OUTPUT_NAME = "submission.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- transforms ----
    tf = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    ])

    # ---- dataset / loader ----
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
        drop_last=False,
    )

    # ---- model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    # ---- inference ----
    rles, filename_and_class = run_inference(model, test_loader, THR)

    # ---- CSV ----
    classes, filenames = zip(*[x.split("_", 1) for x in filename_and_class])
    image_names = [os.path.basename(f) for f in filenames]

    df = pd.DataFrame({
        "image_name": image_names,
        "class": classes,
        "rle": rles,
    })

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    df.to_csv(output_path, index=False)
    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
