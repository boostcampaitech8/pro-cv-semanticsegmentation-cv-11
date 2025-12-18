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
        image = cv2.imread(img_path) / 255.

        label = np.zeros((*image.shape[:2], len(self.classes)), dtype=np.uint8)

        label_path = os.path.join(self.label_root, self.labelnames[idx])
        with open(label_path) as f:
            anns = json.load(f)["annotations"]

        for ann in anns:
            cls = self.class2ind[ann["label"]]
            pts = np.array(ann["points"])
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            label[..., cls] = mask

        if self.transforms:
            data = {"image": image, "mask": label}
            data = self.transforms(**data)
            image = data["image"]
            label = data["mask"]

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label.transpose(2, 0, 1)).float()

        return image, label


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

        image = cv2.imread(img_path) / 255.

        if self.transforms:
            image = self.transforms(image=image)["image"]

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return image, image_name