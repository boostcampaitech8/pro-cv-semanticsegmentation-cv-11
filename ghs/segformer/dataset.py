# dataset.py
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold


def collect_png_json_pairs(image_root: str, label_root: str):
    """
    image_root: .../train/DCM
    label_root: .../train/outputs_json
    returns: filenames(list), labelnames(list)  (상대 경로)
    """
    pngs = []
    jsons = []

    # 이미지 폴더는 하위까지 다 긁음
    for root, _, files in os.walk(image_root):
        for f in files:
            if f.lower().endswith(".png") or f.lower().endswith(".jpg") or f.lower().endswith(".jpeg"):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, image_root)
                pngs.append(rel)

    # 라벨 폴더도 하위까지 다 긁음
    for root, _, files in os.walk(label_root):
        for f in files:
            if f.lower().endswith(".json"):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, label_root)
                jsons.append(rel)

    # 이름 기반 매칭: 확장자 제거하고 키로 맞춤
    def key_no_ext(p):
        base = os.path.basename(p)
        return os.path.splitext(base)[0]

    png_map = {key_no_ext(p): p for p in pngs}
    json_map = {key_no_ext(j): j for j in jsons}

    keys = sorted(list(set(png_map.keys()) & set(json_map.keys())))
    paired_pngs = [png_map[k] for k in keys]
    paired_jsons = [json_map[k] for k in keys]
    return paired_pngs, paired_jsons


class XRayDataset(Dataset):
    """
    - sigmoid/multilabel 관점(one-hot GT, (B,C,H,W))
    - GroupKFold split (폴더 기준 group)
    """
    def __init__(
        self,
        image_root: str,
        label_root: str,
        classes: list,
        pngs: list,
        jsons: list,
        fold: int = 0,
        n_splits: int = 5,
        is_train: bool = True,
        transforms=None,
    ):
        self.image_root = image_root
        self.label_root = label_root
        self.classes = classes
        self.class2ind = {c: i for i, c in enumerate(classes)}
        self.transforms = transforms
        self.is_train = is_train

        filenames = np.array(pngs)
        labelnames = np.array(jsons)

        # group: 상위 폴더 기준(환자/케이스 폴더라 가정)
        groups = np.array([os.path.dirname(f) for f in filenames])
        ys = np.zeros(len(filenames))

        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(filenames, ys, groups))

        train_idx, val_idx = None, None
        for i, (_, v_idx) in enumerate(splits):
            if i == fold:
                val_idx = v_idx
                break

        all_idx = np.arange(len(filenames))
        train_idx = np.setdiff1d(all_idx, val_idx)

        use_idx = train_idx if is_train else val_idx
        self.filenames = filenames[use_idx].tolist()
        self.labelnames = labelnames[use_idx].tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_rel = self.filenames[idx]
        json_rel = self.labelnames[idx]

        img_path = os.path.join(self.image_root, img_rel)
        label_path = os.path.join(self.label_root, json_rel)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(label_path, "r") as f:
            ann = json.load(f)

        h, w = image.shape[:2]
        C = len(self.classes)

        # (C,H,W)로 먼저 만들고, 마지막에 (H,W,C)로 변환해서 albu에 넣음
        mask_chw = np.zeros((C, h, w), dtype=np.uint8)

        for a in ann.get("annotations", []):
            c = a.get("label", None)
            if c not in self.class2ind:
                continue
            ci = self.class2ind[c]
            pts = np.array(a["points"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask_chw[ci], [pts], 1)

        mask = mask_chw.transpose(1, 2, 0)  # (H,W,C)

        if self.transforms is not None:
            out = self.transforms(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]

        # ToTensorV2를 쓰면 이미 torch.Tensor로 오기도 함.
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        else:
            # ToTensorV2(transpose_mask=True)면 (C,H,W)로 바로 옴
            if mask.ndim == 3 and mask.shape[0] != C:
                mask = mask.permute(2, 0, 1).float()
            else:
                mask = mask.float()

        return image, mask


class XRayInferenceDataset(Dataset):
    def __init__(self, image_root: str, pngs: list, transforms=None):
        self.image_root = image_root
        self.filenames = list(pngs)
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_rel = self.filenames[idx]
        img_path = os.path.join(self.image_root, img_rel)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            out = self.transforms(image=image)
            image = out["image"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, img_rel
