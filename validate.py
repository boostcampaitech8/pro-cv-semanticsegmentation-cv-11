import os
import torch
import argparse
import json
import numpy as np
import os.path as osp
import albumentations as A
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import XRayDataset, CLASSES

# TTA 사용 시에만 import
try:
    import ttach as tta
    from ttach import HorizontalFlip
    TTA_AVAILABLE = True
except ImportError:
    TTA_AVAILABLE = False


def unwrap_for_infer(out):
    # DeepSup이면 d1만 사용
    return out[0] if isinstance(out, (tuple, list)) else out


def calculate_dice_score(pred, target, smooth=1e-5):
    """
    Dice score 계산
    pred: (H, W) binary mask
    target: (H, W) binary mask
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def validate(args, data_loader, model, class_thresholds=None, use_tta=False):
    """
    Validation 수행 및 Dice score 계산
    
    Args:
        args: argparse arguments
        data_loader: validation data loader
        model: 학습된 모델
        class_thresholds: 클래스별 threshold 딕셔너리 (선택사항)
        use_tta: TTA 사용 여부
    
    Returns:
        avg_dice: 평균 Dice score
        dices_per_class: 클래스별 Dice score 딕셔너리
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # TTA 사용 시 설정
    if use_tta:
        if not TTA_AVAILABLE:
            raise ImportError("ttach library is not installed. Please install it with: pip install ttach")
        
        # TTA 적용
        transforms = tta.Compose([
            HorizontalFlip(),
        ])
        tta_model = tta.SegmentationTTAWrapper(
            model,
            transforms,
            merge_mode='mean'
        )
        inference_model = tta_model
        desc = "[Validation with TTA...]"
    else:
        inference_model = model
        desc = "[Validation...]"
    
    # 클래스별 threshold tensor 생성
    threshold_tensor = None
    if class_thresholds is not None:
        num_classes = len(CLASSES)
        thresholds = []
        for c in range(num_classes):
            class_name = CLASSES[c]
            thr = class_thresholds.get(class_name, args.thr)
            thresholds.append(thr)
        threshold_tensor = torch.tensor(thresholds, device=device, dtype=torch.float32).view(1, num_classes, 1, 1)
    
    # Dice score 저장용
    dices_per_class = {class_name: [] for class_name in CLASSES}
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc=desc, disable=False) as pbar:
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)  # (B, C, H, W)
                
                # 추론
                if use_tta:
                    outputs = inference_model(images)  # (B, C, H, W) - logit space
                    outputs = torch.sigmoid(outputs)
                else:
                    outputs = inference_model(images)
                    outputs = unwrap_for_infer(outputs)
                    outputs = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    outputs = torch.sigmoid(outputs)
                
                # Threshold 적용
                if threshold_tensor is not None:
                    outputs = (outputs > threshold_tensor).float()
                else:
                    outputs = (outputs > args.thr).float()
                
                # 배치 내 각 샘플에 대해 Dice score 계산
                for i in range(outputs.shape[0]):
                    pred = outputs[i].detach().cpu().numpy()  # (C, H, W)
                    target = labels[i].detach().cpu().numpy()  # (C, H, W)
                    
                    # 각 클래스별 Dice score 계산
                    for c in range(len(CLASSES)):
                        pred_mask = pred[c]  # (H, W)
                        target_mask = target[c]  # (H, W)
                        
                        dice = calculate_dice_score(pred_mask, target_mask)
                        dices_per_class[CLASSES[c]].append(dice)
                
                pbar.update(1)
    
    # 클래스별 평균 Dice score 계산
    avg_dices_per_class = {class_name: np.mean(dices) for class_name, dices in dices_per_class.items()}
    
    # 전체 평균 Dice score
    avg_dice = np.mean(list(avg_dices_per_class.values()))
    
    return avg_dice, avg_dices_per_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate model on validation set and calculate Dice scores")
    parser.add_argument("model", type=str, help="Path to the model file (.pt)")
    parser.add_argument("--image_root", type=str, required=True, help="Path to validation images root")
    parser.add_argument("--label_root", type=str, required=True, help="Path to validation labels root")
    parser.add_argument("--thr", type=float, default=0.5, help="Default threshold for all classes")
    parser.add_argument("--thr_dict", type=str, default=None, help="JSON file path for class-specific thresholds")
    parser.add_argument("--resize", type=int, default=1024, help="Size to resize images")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--use_tta", action="store_true", help="Use Test Time Augmentation (TTA)")
    parser.add_argument("--val_fold", type=int, default=0, help="Validation fold number (0-4)")
    args = parser.parse_args()
    
    # 클래스별 threshold 로드
    class_thresholds = None
    if args.thr_dict:
        with open(args.thr_dict, 'r') as f:
            class_thresholds = json.load(f)
        print(f"Class-wise threshold loaded: {class_thresholds}")
    
    # 데이터셋 준비
    # 파일명 수집
    fnames = sorted([
        osp.relpath(osp.join(root, fname), start=args.image_root)
        for root, _, files in os.walk(args.image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    ])
    
    labels = sorted([
        osp.relpath(osp.join(root, fname), start=args.label_root)
        for root, _, files in os.walk(args.label_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".json"
    ])
    
    # Transform 설정
    tf = A.Resize(height=args.resize, width=args.resize)
    
    # Validation dataset 생성
    val_dataset = XRayDataset(
        fnames=fnames,
        labels=labels,
        image_root=args.image_root,
        label_root=args.label_root,
        fold=args.val_fold,
        transforms=[tf],
        is_train=False
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    # 모델 로드
    print(f"Loading model from: {args.model}")
    model = torch.load(args.model)
    
    # Validation 수행
    avg_dice, dices_per_class = validate(args, val_loader, model, class_thresholds, use_tta=args.use_tta)
    
    # 결과 출력
    print("\n" + "="*50)
    print("Validation Results")
    print("="*50)
    print(f"\nAverage Dice Score: {avg_dice:.4f}\n")
    print("Class-wise Dice Scores:")
    print("-" * 50)
    
    for class_name in CLASSES:
        dice_score = dices_per_class[class_name]
        print(f"{class_name:20s}: {dice_score:.4f}")
    
    print("="*50)

