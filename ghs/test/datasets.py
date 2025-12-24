import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

class XRayDataset(Dataset):
    def __init__(self, image_root, label_root, classes, pngs, jsons, is_train=True, transforms=None):
        self.image_root = image_root
        self.label_root = label_root
        self.classes = classes
        self.class2ind = {v: i for i, v in enumerate(classes)}
        self.transforms = transforms
        self.is_train = is_train

        filenames = np.array(pngs)
        labelnames = np.array(jsons)

        groups = [os.path.dirname(f) for f in filenames]
        ys = [0] * len(filenames)

        gkf = GroupKFold(n_splits=5)

        self.filenames = []
        self.labelnames = []

        for i, (_, val_idx) in enumerate(gkf.split(filenames, ys, groups)):
            if is_train and i == 0:
                continue
            if not is_train and i != 0:
                continue

            self.filenames += list(filenames[val_idx])
            self.labelnames += list(labelnames[val_idx])

            if not is_train:
                break

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.filenames[idx])
        image = cv2.imread(img_path)  # uint8(0~255)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = np.zeros((*image.shape[:2], len(self.classes)), dtype=np.uint8)

        label_path = os.path.join(self.label_root, self.labelnames[idx])
        with open(label_path) as f:
            anns = json.load(f)["annotations"]

        for ann in anns:
            cls = self.class2ind[ann["label"]]
            pts = np.array(ann["points"], dtype=np.int32)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            label[..., cls] = mask

        if self.transforms:
            out = self.transforms(image=image, mask=label)
            image, label = out["image"], out["mask"]  # ToTensorV2면 이미 torch.Tensor

        # ✅ 여기서 torch 변환/transpose 하지 않음!
        return image.float(), label.float()



class XRayInferenceDataset(Dataset):
    def __init__(
        self,
        image_root,
        pngs,
        transforms=None,
    ):
        self.image_root = image_root
        self.filenames = np.array(pngs)
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_name = self.filenames[idx]
        img_path = os.path.join(self.image_root, image_name)

        # ✅ uint8(0~255)로 읽기 (Normalize와 궁합)
        image = cv2.imread(img_path)

        # ✅ (권장) BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            # ✅ ToTensorV2가 포함되어 있으면 image는 이미 torch.Tensor (C,H,W)
            image = self.transforms(image=image)["image"]
            # 안전하게 float 보장
            if isinstance(image, torch.Tensor):
                image = image.float()
            return image, image_name

        # transforms를 안 쓰는 경우 fallback (numpy → torch)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return image, image_name