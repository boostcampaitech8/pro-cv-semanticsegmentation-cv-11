import os
import sys
import os.path as osp
import json
import numpy as np
import pandas as pd
import cv2
import argparse
from tqdm.auto import tqdm

# 프로젝트 루트를 path에 추가
sys.path.append('/data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11')

from dataset import CLASSES

def rle_decode(rle, shape):
    """
    RLE를 mask로 디코딩
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


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


def validate_ensemble(csv_path, label_root, image_root, target_size=(2048, 2048)):
    """
    앙상블 결과 CSV와 GT를 비교하여 Dice score 계산
    
    Args:
        csv_path: 앙상블 결과 CSV 파일 경로
        label_root: GT label JSON 파일 루트 경로
        image_root: 이미지 파일 루트 경로 (이미지 크기 확인용)
        target_size: 예측 결과의 크기 (기본값: 2048x2048)
    
    Returns:
        avg_dice: 평균 Dice score
        dices_per_class: 클래스별 Dice score 딕셔너리
    """
    # CSV 파일 로드
    print(f"Loading ensemble results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 이미지 크기 확인 (첫 번째 이미지로)
    first_image_name = df.iloc[0]['image_name']
    first_image_path = None
    for root, _, files in os.walk(image_root):
        for fname in files:
            if fname == first_image_name:
                first_image_path = osp.join(root, fname)
                break
        if first_image_path:
            break
    
    if first_image_path:
        img = cv2.imread(first_image_path)
        original_size = (img.shape[0], img.shape[1])
    else:
        print("Warning: Could not find image file. Using default size 2048x2048")
        original_size = target_size
    
    # 클래스별 Dice score 저장용
    dices_per_class = {class_name: [] for class_name in CLASSES}
    
    # 이미지별로 그룹화
    image_groups = df.groupby('image_name')
    
    print(f"\nValidating {len(image_groups)} images...")
    with tqdm(total=len(image_groups), desc="Validating") as pbar:
        for image_name, group in image_groups:
            # GT label 로드
            label_path = None
            for root, _, files in os.walk(label_root):
                for fname in files:
                    if fname == image_name.replace('.png', '.json'):
                        label_path = osp.join(root, fname)
                        break
                if label_path:
                    break
            
            if not label_path or not osp.exists(label_path):
                print(f"Warning: GT label not found for {image_name}, skipping...")
                pbar.update(1)
                continue
            
            # GT label 읽기
            with open(label_path, 'r') as f:
                annotations = json.load(f)
            
            # GT mask 생성
            gt_mask = np.zeros(original_size + (len(CLASSES),), dtype=np.uint8)
            for ann in annotations['annotations']:
                class_name = ann['label']
                if class_name in CLASSES:
                    class_idx = CLASSES.index(class_name)
                    points = np.array(ann['points'])
                    cv2.fillPoly(gt_mask[:, :, class_idx], [points], 1)
            
            # 예측 결과 생성 (RLE 디코딩)
            pred_mask = np.zeros(original_size + (len(CLASSES),), dtype=np.uint8)
            for _, row in group.iterrows():
                class_name = row['class']
                rle = row['rle']
                
                if class_name in CLASSES:
                    class_idx = CLASSES.index(class_name)
                    if pd.notna(rle) and rle.strip():
                        try:
                            decoded = rle_decode(rle, original_size)
                            pred_mask[:, :, class_idx] = decoded
                        except Exception as e:
                            print(f"Warning: Failed to decode RLE for {image_name} {class_name}: {e}")
            
            # 각 클래스별 Dice score 계산
            for c, class_name in enumerate(CLASSES):
                pred_class = pred_mask[:, :, c]
                gt_class = gt_mask[:, :, c]
                
                dice = calculate_dice_score(pred_class, gt_class)
                dices_per_class[class_name].append(dice)
            
            pbar.update(1)
    
    # 클래스별 평균 Dice score 계산
    avg_dices_per_class = {class_name: np.mean(dices) for class_name, dices in dices_per_class.items()}
    
    # 전체 평균 Dice score
    avg_dice = np.mean(list(avg_dices_per_class.values()))
    
    return avg_dice, avg_dices_per_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ensemble results against GT and calculate Dice scores")
    parser.add_argument("csv", type=str, help="Path to ensemble result CSV file")
    parser.add_argument("--label_root", type=str, required=True, help="Path to validation labels root")
    parser.add_argument("--image_root", type=str, required=True, help="Path to validation images root (for size detection)")
    args = parser.parse_args()
    
    # Validation 수행
    avg_dice, dices_per_class = validate_ensemble(
        args.csv,
        args.label_root,
        args.image_root
    )
    
    # 결과 출력
    print("\n" + "="*50)
    print("Ensemble Validation Results")
    print("="*50)
    print(f"\nAverage Dice Score: {avg_dice:.4f}\n")
    print("Class-wise Dice Scores:")
    print("-" * 50)
    
    for class_name in CLASSES:
        dice_score = dices_per_class[class_name]
        print(f"{class_name:20s}: {dice_score:.4f}")
    
    print("="*50)

