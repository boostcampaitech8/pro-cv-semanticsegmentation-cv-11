import numpy as np
from dataset import CLASSES

# 손목 뼈 클래스 정의
WRIST_CLASSES = [
    "Trapezium", "Trapezoid", "Capitate", "Hamate",
    "Scaphoid", "Lunate", "Triquetrum", "Pisiform",
]

def get_wrist_indices(class2ind):
    """손목 뼈 클래스의 인덱스 리스트 반환"""
    return [class2ind[c] for c in WRIST_CLASSES if c in class2ind]

def crop_wrist_roi(image, mask_hwc, wrist_indices, min_size=128, margin_frac=0.1):
    """
    손목 뼈 영역만 crop하는 함수
    
    Args:
        image: (H, W, 3) numpy array
        mask_hwc: (H, W, C) numpy array
        wrist_indices: 손목 클래스 인덱스 리스트
        min_size: 최소 크기 (너무 작으면 crop 안 함)
        margin_frac: bbox 주변 margin 비율
    
    Returns:
        image_crop: crop된 이미지
        mask_crop: crop된 마스크
    """
    H, W = image.shape[:2]

    if len(wrist_indices) == 0:
        return image, mask_hwc

    # (H, W, C) → (C, H, W)
    mask_chw = mask_hwc.transpose(2, 0, 1)
    wrist_mask = mask_chw[wrist_indices, ...]  # (num_wrist, H, W)

    # 손목 클래스 전체 OR
    union = wrist_mask.sum(axis=0) > 0  # (H, W), bool

    if not union.any():
        return image, mask_hwc

    # Bounding box 계산
    ys, xs = np.where(union)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # 최소 크기 체크
    if (y_max - y_min + 1) < min_size or (x_max - x_min + 1) < min_size:
        return image, mask_hwc

    # Margin 추가
    h_box = y_max - y_min + 1
    w_box = x_max - x_min + 1
    dy = int(h_box * margin_frac)
    dx = int(w_box * margin_frac)

    ya = max(0, y_min - dy)
    yb = min(H, y_max + dy)
    xa = max(0, x_min - dx)
    xb = min(W, x_max + dx)

    # Crop
    image_crop = image[ya:yb, xa:xb]
    mask_crop = mask_hwc[ya:yb, xa:xb]

    return image_crop, mask_crop

