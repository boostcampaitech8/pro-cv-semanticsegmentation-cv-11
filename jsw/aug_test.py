# 251216
# albumentations aug + batch 단위로 CutMix 적용 aug 공부 코드
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# ------------------------
# 1️⃣ CutMix 함수
# ------------------------
def cutmix(images, masks, alpha=1.0):
    B, C, H, W = images.size()
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(B).to(images.device)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]
    masks[:, :, y1:y2, x1:x2] = masks[rand_index, :, y1:y2, x1:x2]

    return images, masks

# ------------------------
# 2️⃣ Albumentations Transform
# ------------------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# ------------------------
# 3️⃣ Dataset
# ------------------------
class SegDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            # mask 채널 순서 맞추기: [H,W,C] → [C,H,W]
            if mask.ndim == 3:
                mask = mask.permute(2,0,1)

        return image, mask

# ------------------------
# 4️⃣ 간단한 학습 루프
# ------------------------
# 더미 데이터 (예시)
num_samples = 16
images = np.random.randint(0, 256, (num_samples, 256, 256, 3), dtype=np.uint8)
masks = np.random.randint(0, 2, (num_samples, 256, 256, 1), dtype=np.uint8)

dataset = SegDataset(images, masks, transform=train_transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Conv2d(3, 1, kernel_size=3, padding=1).to(device)  # 예시 모델
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for images, masks in loader:
    images, masks = images.to(device), masks.to(device)

    # 50% 확률로 CutMix 적용
    if np.random.rand() > 0.5:
        images, masks = cutmix(images, masks)

    outputs = model(images)
    loss = criterion(outputs, masks.float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Batch loss:", loss.item())
