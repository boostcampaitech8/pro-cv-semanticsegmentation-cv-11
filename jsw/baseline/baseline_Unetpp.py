### FCN ###


''' 1. [Imports & Global Constants](#Imports-&-Global-Constants): 학습에 필요한 라이브러리들을 임포트 하고 미션 안에서 이루어지는 학습 전반을 컨트롤 하기 위한 파라미터들을 설정합니다. '''

#region 라이브러리들 import
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A
import segmentation_models_pytorch as smp

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# visualization
import matplotlib.pyplot as plt

# wandb
from dotenv import load_dotenv
import wandb
load_dotenv() # .env 파일 읽기
wandb.login() # wandb 로그인
#endregion

#region 클래스 정보
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}
#endregion

#region 데이터 경로
IMAGE_ROOT = "/data/ephemeral/home/dataset/train/DCM"
LABEL_ROOT = "/data/ephemeral/home/dataset/train/outputs_json"
#endregion

#region 하이퍼파라미터들 설정
BATCH_SIZE = 8
LR = 1e-4

RANDOM_SEED = 21
NUM_EPOCHS = 1 # 5
VAL_EVERY = 1 # 5

SAVED_DIR = "checkpoints"
if not os.path.exists(SAVED_DIR):                                                          
    os.makedirs(SAVED_DIR)

WANDB_SUBNAME = "Unetpp_res50_loss_edited_5"
CHECKPOINT_NAME = 'Unetpp_res50_loss_edited_5.pt'

CSV_NAME = 'Unetpp_res50_loss_edited_5.csv'
#endregion 


'''2. [Check the Size of the Dataset](#Check-the-Size-of-the-Dataset): 학습에 사용될 데이터가 잘 준비되어있는지 확인합니다. '''

#region .png 및 .json 찾기
# `IMAGE_ROOT` 아래에 있는 모든 폴더를 재귀적으로 순회하면서 확장자가 `.png`인 파일들을 찾습니다.
pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT) # start 기준 상대 경로로 변환 # root : 현재 디렉토리 경로
    for root, _dirs, files in os.walk(IMAGE_ROOT) # IMAGE_ROOT부터 모든 하위 디렉토리를 재귀적으로 탐색
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png" # [1] -> 확장자가 .png
}

# 예시.
# pngs = {
#     "train/0001.png",
#     "train/0002.png",
#     "val/a/001.png",
#     "test/b/xray_03.png"
# }

# 마찬가지로 `LABEL_ROOT` 아래에 있는 모든 폴더를 재귀적으로 순회하면서 확장자가 `.json`인 파일들을 찾습니다.
jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

# 모든 `.png` 파일에 대해 `.json` pair가 존재하는지 체크합니다. 파일명에서 확장자를 제거한 set을 생성하고 두 집합의 차집합의 크기가 0인지 확인합니다.
jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

# 모든 `.png` 파일에 대해 label이 존재하는 것을 확인했습니다. 이름 순으로 정렬해서 짝이 맞도록 합니다.
pngs = sorted(pngs)
jsons = sorted(jsons)
# endregion


''' 3. [Define Dataset Class](#Define-Dataset-Class): 데이터를 원하는 형태로 불러오기 위한 Dataset 클래스를 정의하고, validation을 위한 데이터 스플릿을 진행합니다. '''

#region Custom Dataset Class 정의하기
class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1) # polygon 윤곽선 -> dense mask
            # polygon 윤곽선 -> 데이터 증강 -> dense mask
            # polygon 윤곽선 -> dense mask -> 데이터 증강
            label[..., class_ind] = class_label

        # 지금 보면, 데이터 증강이 fillPoly 이후, 즉 dense mask로 annotation이 변환된 이후에 적용되고 있음.
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # channel first 포맷으로 변경합니다.
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label
#endregion


''' 4. [Check Data Sample](#Check-Data-Sample): 제공된 데이터가 어떤 모습인지 확인합니다.'''

# 시각화는 생략
#region Dataset 설정
tf = A.Resize(512, 512)
train_dataset = XRayDataset(is_train=True, transforms=tf)
valid_dataset = XRayDataset(is_train=False, transforms=tf)
#endregion


''' 5. [Setup Dataloader](#Setup-Dataloader): 학습을 위해 데이터를 배치로 불러오기 위한 Dataloader를 만듭니다. '''

#region Dataloader 정의하기 
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)

# 주의: validation data는 이미지 크기가 크기 때문에 'num_wokers'는 커지면 메모리 에러가 발생할 수 있습니다.
valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=8,
    shuffle=False,
    num_workers=0, # num_workers=0 -> 메인 프로세스에서 데이터를 동기적으로 로딩, 이미지 크기 크면 0 또는 1~2 권장
    drop_last=False
)
#endregion


''' 6. [Define Functions for Training](#Define-Functions-for-Training): 학습을 도와주는 함수들을 정의합니다.'''

#region utils
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, file_name=CHECKPOINT_NAME):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    # os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED) # 대부분은 필요 x.
# endregion

#region validation()
# 참고)
# images : (B, C, H, W)
    # B = batch size, C = 3 (RGB), H,W = 이미지 크기
# masks : (B, num_classes, H, W)
    # multi-channel 0/1 mask
    # num_classes = len(CLASSES)
# def validation(epoch, model, data_loader, criterion, thr=0.5):
#     print(f'Start validation #{epoch:2d}')
#     model.eval()
#     model = model.cuda()

#     dices = []
#     with torch.no_grad():
#         n_class = len(CLASSES)
#         total_loss = 0
#         cnt = 0

#         for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
#             images, masks = images.cuda(), masks.cuda()         
            
#             outputs = model(images)
            
#             output_h, output_w = outputs.size(-2), outputs.size(-1)
#             mask_h, mask_w = masks.size(-2), masks.size(-1)
            
#             # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
#             if output_h != mask_h or output_w != mask_w:
#                 outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
#             loss = criterion(outputs, masks)
#             total_loss += loss
#             cnt += 1
            
#             outputs = torch.sigmoid(outputs)
#             outputs = (outputs > thr).detach()
#             masks = masks.detach()
            
#             dice = dice_coef(outputs, masks)
#             dices.append(dice.cpu())
                
#     dices = torch.cat(dices, 0)
#     dices_per_class = torch.mean(dices, 0)
#     dice_str = [
#         f"{c:<12}: {d.item():.4f}"
#         for c, d in zip(CLASSES, dices_per_class)
#     ]
#     dice_str = "\n".join(dice_str)
#     print(dice_str)
    
#     avg_dice = torch.mean(dices_per_class).item()
    
#     return avg_dice

def soft_dice(y_true, y_prob, eps=1e-6):
    # y_true, y_prob: (B,C,H,W) float
    inter = (y_true * y_prob).sum(dim=(2,3))
    union = y_true.sum(dim=(2,3)) + y_prob.sum(dim=(2,3))
    dice  = (2*inter + eps) / (union + eps)
    return dice.mean().item() if dice.numel() > 0 else 0.0

def validation(epoch, model, loader, criterion, thrs=(0.3,0.4,0.5)):
    model.eval()
    loss_sum, n = 0.0, 0
    dice_thr_sum = {t:0.0 for t in thrs}
    dice_soft_sum = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.cuda(non_blocking=True)
            masks  = masks.cuda(non_blocking=True).float()  # BCE류/ Dice류 모두 float 필요

            logits = model(images)  # SMP는 tensor logits 반환

            # 251218 추가: 없으니까 이미지 사이즈 차이 에러 남.
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            if logits.size(-2) != mask_h or logits.size(-1) != mask_w:
                logits = F.interpolate(logits, size=(mask_h, mask_w), mode="bilinear", align_corners=False)

            loss = criterion(logits, masks)
            loss_sum += loss.item(); n += 1

            prob = torch.sigmoid(logits)
            dice_soft_sum += soft_dice(masks, prob)

            for t in thrs:
                pred = (prob > t).float()
                dice_thr_sum[t] += soft_dice(masks, pred)

    print(f"Epoch[{epoch}] val_loss={loss_sum/n:.4f}, soft_dice={dice_soft_sum/n:.4f}, " +
          ", ".join([f"dice@{t}={dice_thr_sum[t]/n:.4f}" for t in thrs]))
    return dice_thr_sum[0.5]/n


#endregion

#region train()
def train(model, data_loader, val_loader, criterion, optimizer):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.
    
    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            outputs = model(images)
            
            # loss를 계산합니다.
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
            ### 완디비 train 설정 by 형석님 ###
                wandb.log({
                    "train/loss": loss.item(),
                    "epoch": epoch + 1,
                    "step": epoch * len(data_loader) + step
                })
        
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        # best model 선정 기준: dice
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            
            ### 완디비 val 설정 by 형석님 ###
            wandb.log({
                "val/mean_dice": dice,
                "epoch": epoch + 1
            })

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)
#endregion


''' 7. [Training](#Training): 학습을 진행합니다.'''

#region 모델 설정
# model = models.segmentation.fcn_resnet50(pretrained=True)
# torchvision에서 제공하는 semantic segmentation 모델 모듈에서,
# ResNet-50을 backbone(encoder)으로 쓰는 FCN 모델 가져옴.
# pretrained=True -> ImageNet으로 pre-trained된 ResNet-50 백본을 사용함.

# output class 개수를 dataset에 맞도록 수정합니다.
# model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

# 모델을 Unet으로 변환

# model = smp.UnetPlusPlus(
#     encoder_name="resnet50",      # backbone
#     encoder_weights="imagenet",   # pretrained weights
#     in_channels=3,                # RGB 이미지
#     classes=len(CLASSES),         # 출력 클래스 수
# )

import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4", 
    encoder_weights="imagenet",
    in_channels=3,                  # 입력 이미지 채널 (RGB)
    classes=len(CLASSES)            # 출력 채널 (29개 뼈)
)

model = model.cuda()
#endregion

#region loss 및 optimizer 설정
# Loss function을 정의합니다.
# def dice_loss(pred, target, smooth = 1.):
#     pred = pred.contiguous()
#     target = target.contiguous()   
#     intersection = (pred * target).sum(dim=2).sum(dim=2)
#     loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))

#     return loss.mean()

# def calc_loss(pred, target, bce_weight = 0.5):
#     bce = F.binary_cross_entropy_with_logits(pred, target)
#     pred = F.sigmoid(pred)
#     dice = dice_loss(pred, target)
#     loss = bce * bce_weight + dice * (1 - bce_weight)

#     return loss

# bce_loss = nn.BCEWithLogitsLoss()
# dice_loss = DiceLoss(mode="binary", from_logits=True)

# def calc_loss(outputs, targets):
#     return bce_loss(outputs, targets) + dice_loss(outputs, targets)

# criterion = calc_loss
dice = smp.losses.DiceLoss(mode="multilabel", from_logits=True)
bce  = smp.losses.SoftBCEWithLogitsLoss()

class ComboLoss(nn.Module):
    def __init__(self, w_dice=0.7, w_bce=0.3):
        super().__init__()
        self.dice = dice
        self.bce = bce
        self.w_dice = w_dice
        self.w_bce = w_bce
    def forward(self, y_pred, y_true):
        return self.w_dice * self.dice(y_pred, y_true) + self.w_bce * self.bce(y_pred, y_true)
criterion = ComboLoss() 

# criterion = nn.BCEWithLogitsLoss() # BCE + Sigmoid 연산 포함
# Label format: (batch, num_classes, H, W)
# 각 클래스 채널이 0/1 binary mask니까 BCE를 쓴 것.

# Optimizer를 정의합니다.
optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)

# 시드를 설정합니다.
set_seed()
#endregion

#region wandb init
### wandb 초기화 by 형석님 ###
wandb.init(
    entity="cv_11",
    project="cv-11-SEG",
    name=f"jsw-{WANDB_SUBNAME}-{model.__class__.__name__}_epochs{NUM_EPOCHS}",
    config={
        "optimizer": optimizer.__class__.__name__,
        "loss": criterion.__class__.__name__,
        "lr": LR,
        "epochs": NUM_EPOCHS,
        "model": model.__class__.__name__,
    }
)
#endregion

#region 학습 돌리기
train(model, train_loader, valid_loader, criterion, optimizer)
wandb.finish()
#endregion


''' 8. [Inference](#Inference): 인퍼런스에 필요한 함수들을 정의하고, 인퍼런스를 진행합니다.'''

#region inference 모델 및 데이터 불러오기
model = torch.load(os.path.join(SAVED_DIR, CHECKPOINT_NAME))
IMAGE_ROOT = "../../dataset/test/DCM"
pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}
#endregion

#region utils
# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE로 인코딩된 결과를 mask map으로 복원합니다.
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)
#endregion

#region Custom Dataset Class 정의하기
class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  
        
        image = torch.from_numpy(image).float()
            
        return image, image_name
#endregion

#region test()
def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)
            
            # upscaling
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class
#endregion

#region data augmentation 정의
tf = A.Resize(512, 512)
#endregion

#region Dataset 설정 및 Dataloader 정의
test_dataset = XRayInferenceDataset(transforms=tf)
test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)
#endregion

#region 추론 돌리기
rles, filename_and_class = test(model, test_loader)
#endregion


''' 9. [Result Visualization](#Result-Visualization): 인퍼런스 결과를 확인해봅니다.'''

# 생략

''' 10. [To CSV](#To-CSV): 인퍼런스 결과를 제출을 위한 포맷으로 변경합니다.'''

#region .csv 파일로 저장하기
classes, filename = zip(*[x.split("_") for x in filename_and_class])
image_name = [os.path.basename(f) for f in filename]
df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})
if not os.path.exists("outputs"):                                                          
    os.makedirs("outputs")
df.to_csv(f'outputs/{CSV_NAME}', index=False)
print(f'.csv file saved')
#endregion