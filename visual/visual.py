import os
import cv2
import argparse
import numpy as np
import pandas as pd


# -----------------------------
# RLE decode
# -----------------------------
def decode_rle_to_mask(rle, height, width):
    if rle == "" or pd.isna(rle):
        return np.zeros((height, width), dtype=np.uint8)

    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths

    mask = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1

    return mask.reshape(height, width)


# -----------------------------
# util: ID 폴더에서 이미지 자동 수집
# -----------------------------
def collect_images_from_ids(image_root, id_list):
    image_names = []

    for cur_id in id_list:
        id_dir = os.path.join(image_root, cur_id)
        if not os.path.isdir(id_dir):
            print(f"⚠ ID folder not found: {id_dir}")
            continue

        for fname in sorted(os.listdir(id_dir)):
            if fname.lower().endswith(".png"):
                image_names.append((cur_id, fname))

    return image_names


# -----------------------------
# main visualization
# -----------------------------
def visualize_images(
    csv_path,
    image_root,
    image_pairs,   # (id, image_name)
    save_root,
    only_classes=None,
    height=2048,
    width=2048,
    alpha=0.4,
):

    df = pd.read_csv(csv_path)
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    save_root = os.path.join(save_root, csv_name)
    os.makedirs(save_root, exist_ok=True)

    if only_classes is not None:
        df = df[df["class"].isin(only_classes)]

    for cur_id, image_name in image_pairs:
        print(f"[Visualizing] {cur_id}/{image_name}")

        df_img = df[df["image_name"] == image_name]
        if df_img.empty:
            print(f"  ⚠ No prediction for {image_name}")
            continue

        img_path = os.path.join(image_root, cur_id, image_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"  ❌ Image load failed: {img_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        save_dir = os.path.join(
            save_root, cur_id, os.path.splitext(image_name)[0]
        )
        os.makedirs(save_dir, exist_ok=True)

        for _, row in df_img.iterrows():
            cls = row["class"]
            rle = row["rle"]

            mask = decode_rle_to_mask(rle, height, width)

            overlay = image.copy()
            overlay[mask == 1] = [255, 0, 0]

            blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            concat = np.concatenate([image, blended], axis=1)

            out_path = os.path.join(save_dir, f"{cls}.png")
            cv2.imwrite(out_path, cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)

    parser.add_argument(
        "--ids",
        nargs="+",
        required=True,
        help="ID folders to visualize (e.g. ID040 ID041)",
    )

    parser.add_argument(
        "--only-class",
        nargs="+",
        default=None,
        help="Only visualize these classes (e.g. Pisiform Triquetrum)",
    )

    parser.add_argument("--save_root", type=str, default="./vis_results")
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=0.4)

    args = parser.parse_args()

    image_pairs = collect_images_from_ids(
        image_root=args.image_root,
        id_list=args.ids,
    )

    if len(image_pairs) == 0:
        raise RuntimeError("No images found in given IDs")

    visualize_images(
        csv_path=args.csv,
        image_root=args.image_root,
        image_pairs=image_pairs,
        save_root=args.save_root,
        only_classes=args.only_class,
        height=args.height,
        width=args.width,
        alpha=args.alpha,
    )
