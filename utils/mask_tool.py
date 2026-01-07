# python mask_tool.py --mask_create
# python mask_tool.py --mask_delete

import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw


# -----------------------------
# Class definitions
# -----------------------------
CLASS = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19',
    'Trapezium', 'Trapezoid', 'Capitate', 'Hamate',
    'Scaphoid', 'Lunate', 'Triquetrum', 'Pisiform',
    'Radius', 'Ulna',
]

CLASS2IDX = {name: idx + 1 for idx, name in enumerate(CLASS)}


# -----------------------------
# Mask creation
# -----------------------------
def create_mask_img_json(
    source_image_dir,
    source_json_dir,
    target_classes=None,
):
    if target_classes is None:
        target_classes = list(CLASS2IDX.values())

    for folder_name in os.listdir(source_image_dir):
        image_folder = os.path.join(source_image_dir, folder_name)
        json_folder = os.path.join(source_json_dir, folder_name)

        if not os.path.isdir(image_folder):
            continue

        image_files = [
            f for f in os.listdir(image_folder)
            if f.endswith(".png")
        ]

        for filename in tqdm(image_files, desc=f"Processing {folder_name}"):
            image_path = os.path.join(image_folder, filename)
            json_path = os.path.join(
                json_folder,
                filename.replace(".png", ".json")
            )

            if not os.path.exists(json_path):
                continue

            image = cv2.imread(image_path)
            image = image.astype("float32") / 255.0
            original_image = image.copy()

            height, width = image.shape[:2]

            with open(json_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)["annotations"]

            mask = np.zeros((height, width), dtype=np.uint8)
            new_annotations = []

            for ann in annotations:
                class_idx = CLASS2IDX[ann["label"]]
                if class_idx not in target_classes:
                    continue

                points = np.array(ann["points"], dtype=np.int32)

                mask_img = Image.new("L", (width, height), 0)
                ImageDraw.Draw(mask_img).polygon(
                    [tuple(p) for p in points],
                    outline=1,
                    fill=1,
                )

                class_mask = np.array(mask_img, dtype=np.uint8)
                mask = np.maximum(mask, class_mask)
                new_annotations.append(ann)

            masked_image = original_image.copy()
            masked_image[mask == 0] = 0

            masked_filename = filename.replace("image", "masked_image", 1)
            cv2.imwrite(
                os.path.join(image_folder, masked_filename),
                (masked_image * 255).astype(np.uint8),
            )

            with open(
                os.path.join(
                    json_folder,
                    masked_filename.replace(".png", ".json"),
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {"annotations": new_annotations},
                    f,
                    ensure_ascii=False,
                    indent=4,
                )


# -----------------------------
# Mask deletion
# -----------------------------
def delete_mask_img_json(
    source_dir,
    prefix="masked_image",
):
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith(".png") and filename.startswith(prefix):
                os.remove(os.path.join(folder_path, filename))


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask create / delete utility"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--mask_create",
        action="store_true",
        help="Create masked images and JSON files",
    )
    group.add_argument(
        "--mask_delete",
        action="store_true",
        help="Delete masked images",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/data/train/DCM",
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/data/train/outputs_json",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mask_create:
        create_mask_img_json(
            args.image_dir,
            args.json_dir,
        )

    if args.mask_delete:
        delete_mask_img_json(args.image_dir)


if __name__ == "__main__":
    main()
