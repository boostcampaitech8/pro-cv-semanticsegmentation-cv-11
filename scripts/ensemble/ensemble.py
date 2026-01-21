import os
import sys
import os.path as osp

# 프로젝트 루트를 path에 추가 (import 전에 실행)
sys.path.append('/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11')

import torch
import argparse
import json
import time
import numpy as np
import pandas as pd
import albumentations as A
import torch.nn.functional as F
import cv2

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import XRayInferenceDataset, XRayDataset, CLASSES
from models.model_picker import ModelPicker
import ttach as tta
from ttach import HorizontalFlip, Multiply

# Wrist class 정의 (클래스별 선택적 앙상블용)
WRIST_CLASSES = [
    "Trapezium", "Trapezoid", "Capitate", "Hamate",
    "Scaphoid", "Lunate", "Triquetrum", "Pisiform",
]

# Wrist class 인덱스 계산 (전역 변수로 한 번만 계산)
WRIST_INDICES = [CLASSES.index(c) for c in WRIST_CLASSES if c in CLASSES]


def get_wrist_roi_coords(wrist_mask_tensor, original_h, original_w, margin_frac=0.15, threshold=0.5):
    """
    Wrist mask에서 손목 부위 좌표 추출
    
    Args:
        wrist_mask_tensor: (C, H, W) wrist class들의 mask tensor (sigmoid 적용된 probability)
        original_h: 원본 이미지 높이
        original_w: 원본 이미지 너비
        margin_frac: bbox 주변 margin 비율
        threshold: wrist detection threshold
    
    Returns:
        ya, yb, xa, xb: crop 좌표
        found: wrist가 발견되었는지 여부
    """
    # Wrist class들의 heatmap 합산
    wrist_heatmap = wrist_mask_tensor[WRIST_INDICES, :, :].sum(dim=0)  # (H, W)
    
    # Threshold 이상인 영역 찾기
    ys, xs = torch.where(wrist_heatmap > threshold)
    
    if len(ys) == 0:
        return 0, original_h, 0, original_w, False
    
    y_min, y_max = ys.min().item(), ys.max().item()
    x_min, x_max = xs.min().item(), xs.max().item()
    
    # Margin 추가
    h_box = y_max - y_min + 1
    w_box = x_max - x_min + 1
    dy = int(h_box * margin_frac)
    dx = int(w_box * margin_frac)
    
    ya = max(0, y_min - dy)
    yb = min(original_h, y_max + dy)
    xa = max(0, x_min - dx)
    xb = min(original_w, x_max + dx)
    
    return ya, yb, xa, xb, True


def paste_crop_to_original(crop_pred_tensor, original_h, original_w, coords):
    """
    Crop된 예측 결과를 원본 사이즈 캔버스에 복원
    
    Args:
        crop_pred_tensor: (C, H_crop, W_crop) crop된 예측 결과
        original_h: 원본 이미지 높이
        original_w: 원본 이미지 너비
        coords: (ya, yb, xa, xb) crop 좌표
    
    Returns:
        full_canvas: (C, original_h, original_w) 원본 크기의 예측 결과
    """
    ya, yb, xa, xb = coords
    target_h, target_w = yb - ya, xb - xa
    C = crop_pred_tensor.shape[0]
    device = crop_pred_tensor.device
    
    full_canvas = torch.zeros((C, original_h, original_w), device=device, dtype=crop_pred_tensor.dtype)
    
    if target_h > 0 and target_w > 0:
        # Crop 예측을 원본 크기의 해당 영역 크기로 resize
        resized_crop = F.interpolate(
            crop_pred_tensor.unsqueeze(0),  # (1, C, H_crop, W_crop)
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (C, target_h, target_w)
        
        full_canvas[:, ya:yb, xa:xb] = resized_crop
    
    return full_canvas

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


def unwrap_for_infer(out):
    # DeepSup이면 d1만 사용
    return out[0] if isinstance(out, (tuple, list)) else out


class ModelWithPostProcess(torch.nn.Module):
    """모델 출력에 후처리(interpolate)를 적용하는 래퍼"""
    def __init__(self, model, target_size=(2048, 2048)):
        super().__init__()
        self.model = model
        self.target_size = target_size
    
    def forward(self, x):
        out = self.model(x)
        out = unwrap_for_infer(out)  # tensor (B,C,H,W)
        out = F.interpolate(out, size=self.target_size, mode="bilinear")
        # sigmoid는 ttach의 merge 후에 적용 (ttach 기본 방식)
        return out


def load_model(model_path, device, use_tta=False, model_config=None, for_validation=False):
    """
    모델을 로드하고 TTA 래퍼를 생성
    
    Args:
        model_path: 모델 파일 경로
        device: 디바이스
        use_tta: TTA 사용 여부
        model_config: state_dict인 경우 모델 구조 정보 (dict)
                      예: {"model_name": "UnetPlusPlus", "encoder_name": "se_resnext101_32x4d", 
                           "encoder_weights": None, "in_channels": 3, "classes": 29}
        for_validation: validation 모드인 경우 True (ModelWithPostProcess 사용 안 함, validate.py와 동일하게)
    
    Returns:
        tta_model: 추론에 사용할 모델 (TTA 적용 또는 일반)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # 전체 모델이 저장된 경우
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    # state_dict만 저장된 경우
    elif isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # state_dict가 직접 저장된 경우
            state_dict = checkpoint
        
        # 모델 구조 정보가 제공된 경우 모델 생성
        if model_config is not None:
            print(f"  Creating model from config: {model_config}")
            model_picker = ModelPicker()
            model = model_picker.get_model(**model_config)
            
            # state_dict 키가 "model."로 시작하는 경우 제거 (UnetPlusPlus 래퍼 때문)
            if any(k.startswith('model.') for k in state_dict.keys()):
                # "model." 접두사 제거
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict[k[6:]] = v  # "model." 제거
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
        else:
            raise ValueError(
                f"Model file '{model_path}' contains only state_dict. "
                f"Please provide model_config argument with model structure information.\n"
                f"Example: {{'model_name': 'UnetPlusPlus', 'encoder_name': 'se_resnext101_32x4d', "
                f"'encoder_weights': None, 'in_channels': 3, 'classes': 29}}"
            )
    else:
        # 기타 경우: 전체 모델로 간주
        model = checkpoint
    
    model.eval()
    
    # Validation 모드인 경우: validate.py와 동일하게 모델을 직접 사용 (ModelWithPostProcess 사용 안 함)
    if for_validation:
        # TTA 적용 (ttach 기반)
        if use_tta:
            transforms = tta.Compose([
                HorizontalFlip(),
                # Multiply(factors=[0.95, 1.05]),
            ])
            tta_model = tta.SegmentationTTAWrapper(
                model,
                transforms,
                merge_mode='mean'  # 평균
            )
        else:
            tta_model = model
    else:
        # Inference 모드: ModelWithPostProcess로 감싸서 2048x2048로 고정
        model_wrapper = ModelWithPostProcess(model, target_size=(2048, 2048))
        model_wrapper.eval()
        
        # TTA 적용 (ttach 기반)
        if use_tta:
            transforms = tta.Compose([
                HorizontalFlip(),
                # Multiply(factors=[0.95, 1.05]),
            ])
            tta_model = tta.SegmentationTTAWrapper(
                model_wrapper,
                transforms,
                merge_mode='mean'  # 평균
            )
        else:
            tta_model = model_wrapper
    
    return tta_model


def ensemble_inference(args, data_loaders, class_thresholds=None):
    """
    여러 모델을 배치 단위로 앙상블하여 추론 수행 (메모리 효율적)
    각 모델마다 다른 데이터로더 사용 가능
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 모든 모델 로드
    print(f"Loading {len(args.models)} models...")
    models = []
    for i, model_path in enumerate(args.models):
        start_time = time.time()
        print(f"[{i+1}/{len(args.models)}] Loading: {osp.basename(model_path)}")
        
        # 해당 모델의 config 가져오기 (있는 경우)
        model_config = None
        if args.model_configs and i < len(args.model_configs):
            model_config = args.model_configs[i]
        
        tta_model = load_model(model_path, device, use_tta=args.use_tta, model_config=model_config)
        models.append(tta_model)
        print(f"  Model loaded in {time.time() - start_time:.2f}s")
    
    # 가중치 설정
    if args.weights is not None:
        if len(args.weights) != len(args.models):
            raise ValueError(f"Number of weights ({len(args.weights)}) must match number of models ({len(args.models)})")
        weights = np.array(args.weights)
        weights = weights / weights.sum()  # 정규화
        print(f"\nUsing weighted average with weights: {weights}")
    else:
        weights = None
        print(f"\nUsing simple average (equal weights)")
    
    # 클래스 이름 가져오기 (dataset에서 - 모든 데이터로더는 같은 클래스 정보를 가짐)
    ind2class = data_loaders[0].dataset.ind2class
    num_classes = len(ind2class)
    
    # 클래스별 threshold tensor 생성
    threshold_tensor = None
    if class_thresholds is not None:
        thresholds = []
        for c in range(num_classes):
            class_name = ind2class[c]
            thr = class_thresholds.get(class_name, args.thr)
            thresholds.append(thr)
        threshold_tensor = torch.tensor(thresholds, device=device, dtype=torch.float32).view(1, num_classes, 1, 1)
    
    # 배치 단위로 추론 및 앙상블
    # 모든 데이터로더를 동시에 iterate (zip 사용)
    print("\nProcessing batches (ensemble on-the-fly)...")
    rles = []
    filename_and_class = []
    
    # 모든 데이터로더의 iterator 생성
    loader_iters = [iter(loader) for loader in data_loaders]
    num_batches = len(data_loaders[0])
    
    with torch.no_grad():
        with tqdm(total=num_batches, desc="Ensemble inference") as pbar:
            for batch_idx in range(num_batches):
                # 각 데이터로더에서 배치 가져오기
                batch_data = []
                image_names = None
                for loader_iter in loader_iters:
                    images, names = next(loader_iter)
                    batch_data.append(images.to(device))
                    if image_names is None:
                        image_names = names
                
                # 각 모델로 추론 (각 모델은 자신의 데이터로더의 배치 사용)
                model_outputs = []
                
                # Crop 모드일 때 처리
                if args.crop_mode and args.crop_model_idx is not None and args.crop_model_idx < len(models):
                    crop_model_idx = args.crop_model_idx
                    full_model_indices = [i for i in range(len(models)) if i != crop_model_idx]
                    
                    # Full 모델들로 먼저 예측 (wrist ROI 추출용)
                    full_outputs_list = []
                    for i in full_model_indices:
                        outputs = models[i](batch_data[i])
                        outputs = torch.sigmoid(outputs)
                        full_outputs_list.append(outputs)
                    
                    # Full 모델들의 평균으로 wrist ROI 추출 (또는 첫 번째 full 모델 사용)
                    if len(full_outputs_list) > 0:
                        full_pred = torch.mean(torch.stack(full_outputs_list), dim=0)  # (B, C, H, W)
                    else:
                        full_pred = None
                    
                    # 배치 내 각 이미지에 대해 crop 처리
                    batch_size = batch_data[0].shape[0]
                    crop_outputs_list = []
                    
                    for b in range(batch_size):
                        # 원본 이미지 읽기
                        image_name = image_names[b]
                        image_path = osp.join(args.image_root, image_name)
                        ori_img = cv2.imread(image_path)
                        if ori_img is None:
                            # 이미지 읽기 실패 시 full 모델 예측 사용
                            crop_outputs_list.append(full_pred[b:b+1])
                            continue
                        
                        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
                        ori_h, ori_w = ori_img.shape[:2]
                        
                        # 예측 크기 (2048x2048로 interpolate된 크기)
                        pred_h, pred_w = full_pred.shape[2], full_pred.shape[3]
                        
                        # Wrist ROI 좌표 추출 (예측 크기 기준)
                        ya_pred, yb_pred, xa_pred, xb_pred, found = get_wrist_roi_coords(
                            full_pred[b], pred_h, pred_w, margin_frac=0.15, threshold=0.5
                        )
                        
                        if found:
                            # 예측 크기 좌표를 원본 크기로 스케일링
                            scale_h = ori_h / pred_h
                            scale_w = ori_w / pred_w
                            ya = int(ya_pred * scale_h)
                            yb = int(yb_pred * scale_h)
                            xa = int(xa_pred * scale_w)
                            xb = int(xb_pred * scale_w)
                            
                            # 경계 체크
                            ya = max(0, min(ya, ori_h))
                            yb = max(0, min(yb, ori_h))
                            xa = max(0, min(xa, ori_w))
                            xb = max(0, min(xb, ori_w))
                            
                            # Crop
                            crop_img = ori_img[ya:yb, xa:xb]
                            
                            if crop_img.size > 0:
                                # Crop 이미지 전처리 (crop 모델의 resize 크기 사용)
                                # data_loaders에서 crop 모델의 resize 크기 가져오기
                                crop_resize = data_loaders[crop_model_idx].dataset.transforms.transforms[0].height if hasattr(data_loaders[crop_model_idx].dataset.transforms, 'transforms') and len(data_loaders[crop_model_idx].dataset.transforms.transforms) > 0 else 1024
                                crop_tf = A.Compose([
                                    A.Resize(crop_resize, crop_resize),
                                    A.Normalize(),
                                ])
                                crop_transformed = crop_tf(image=crop_img)['image']
                                crop_tensor = torch.from_numpy(crop_transformed.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                                
                                # Crop 모델로 예측
                                crop_output = models[crop_model_idx](crop_tensor)
                                crop_output = torch.sigmoid(crop_output).squeeze(0)  # (C, H_crop, W_crop)
                                
                                # 예측 크기 좌표로 복원 (나중에 interpolate로 원본 크기로 변환됨)
                                crop_output_full = paste_crop_to_original(crop_output, pred_h, pred_w, (ya_pred, yb_pred, xa_pred, xb_pred))
                                crop_outputs_list.append(crop_output_full.unsqueeze(0))
                            else:
                                # Crop 실패 시 full 모델 예측 사용
                                crop_outputs_list.append(full_pred[b:b+1])
                        else:
                            # Wrist ROI를 찾지 못한 경우 full 모델 예측 사용
                            crop_outputs_list.append(full_pred[b:b+1])
                    
                    # Crop 모델 출력을 배치로 합치기
                    crop_outputs = torch.cat(crop_outputs_list, dim=0)  # (B, C, H, W)
                    model_outputs.append(crop_outputs)
                    
                    # Full 모델 출력도 추가
                    for i, full_out in enumerate(full_outputs_list):
                        model_outputs.insert(full_model_indices[i], full_out)
                else:
                    # 일반 모드: 모든 모델에 동일한 이미지 입력
                    for model, images in zip(models, batch_data):
                        outputs = model(images)  # (B, C, 2048, 2048) - logit space
                        outputs = torch.sigmoid(outputs)  # sigmoid 적용
                        model_outputs.append(outputs)
                
                # 가중 평균으로 앙상블 (클래스별 선택적 앙상블 지원)
                if args.crop_model_idx is not None and args.crop_model_idx < len(models):
                    # 클래스별 선택적 앙상블: Wrist class는 crop 모델만, 나머지는 full 모델들만
                    ensemble_outputs = torch.zeros_like(model_outputs[0])
                    crop_model_idx = args.crop_model_idx
                    full_model_indices = [i for i in range(len(models)) if i != crop_model_idx]
                    
                    # Wrist class 인덱스 계산
                    wrist_indices = [CLASSES.index(c) for c in WRIST_CLASSES if c in CLASSES]
                    
                    if weights is not None:
                        # Full 모델들의 가중치 합 계산
                        full_weights_sum = sum(weights[i] for i in full_model_indices)
                        
                        for c in range(ensemble_outputs.shape[1]):  # 각 클래스별로
                            if c in wrist_indices:
                                # Wrist class: crop 모델만 사용
                                ensemble_outputs[:, c, :, :] = model_outputs[crop_model_idx][:, c, :, :]
                            else:
                                # 나머지 클래스: full 모델들만 앙상블 (가중치 정규화)
                                if full_weights_sum > 0:
                                    for i in full_model_indices:
                                        normalized_weight = weights[i] / full_weights_sum
                                        ensemble_outputs[:, c, :, :] += normalized_weight * model_outputs[i][:, c, :, :]
                                else:
                                    # 가중치가 0인 경우 단순 평균
                                    full_outputs = torch.stack([model_outputs[i][:, c, :, :] for i in full_model_indices])
                                    ensemble_outputs[:, c, :, :] = torch.mean(full_outputs, dim=0)
                    else:
                        # 단순 평균 버전
                        for c in range(ensemble_outputs.shape[1]):
                            if c in wrist_indices:
                                # Wrist class: crop 모델만 사용
                                ensemble_outputs[:, c, :, :] = model_outputs[crop_model_idx][:, c, :, :]
                            else:
                                # 나머지 클래스: full 모델들만 평균
                                full_outputs = torch.stack([model_outputs[i][:, c, :, :] for i in full_model_indices])
                                ensemble_outputs[:, c, :, :] = torch.mean(full_outputs, dim=0)
                else:
                    # 기존 방식: 모든 클래스에 대해 일반 앙상블
                    if weights is not None:
                        # 가중 평균
                        ensemble_outputs = torch.zeros_like(model_outputs[0])
                        for i, output in enumerate(model_outputs):
                            ensemble_outputs += weights[i] * output
                    else:
                        # 단순 평균
                        ensemble_outputs = torch.mean(torch.stack(model_outputs), dim=0)
                
                # Threshold 적용
                if threshold_tensor is not None:
                    thresholded = (ensemble_outputs > threshold_tensor).detach().cpu().numpy()
                else:
                    thresholded = (ensemble_outputs > args.thr).detach().cpu().numpy()
                
                # RLE 인코딩
                for output, image_name in zip(thresholded, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{ind2class[c]}_{image_name}")
                
                pbar.update(1)
    
    return rles, filename_and_class


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


def ensemble_validate(args, data_loaders, val_loader, class_thresholds=None):
    """
    앙상블 결과를 바로 GT와 비교하여 Dice score 계산 (CSV 저장 없음)
    
    Args:
        args: argparse arguments
        data_loaders: 각 모델용 데이터로더 리스트
        val_loader: validation 데이터로더 (GT 포함)
        class_thresholds: 클래스별 threshold 딕셔너리 (선택사항)
    
    Returns:
        avg_dice: 평균 Dice score
        dices_per_class: 클래스별 Dice score 딕셔너리
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 모든 모델 로드 (validation 모드: ModelWithPostProcess 사용 안 함, validate.py와 동일하게)
    print(f"Loading {len(args.models)} models...")
    models = []
    for i, model_path in enumerate(args.models):
        start_time = time.time()
        print(f"[{i+1}/{len(args.models)}] Loading: {osp.basename(model_path)}")
        
        # 해당 모델의 config 가져오기 (있는 경우)
        model_config = None
        if args.model_configs and i < len(args.model_configs):
            model_config = args.model_configs[i]
        
        tta_model = load_model(model_path, device, use_tta=args.use_tta, model_config=model_config, for_validation=True)
        models.append(tta_model)
        print(f"  Model loaded in {time.time() - start_time:.2f}s")
    
    # 가중치 설정
    if args.weights is not None:
        if len(args.weights) != len(args.models):
            raise ValueError(f"Number of weights ({len(args.weights)}) must match number of models ({len(args.models)})")
        weights = np.array(args.weights)
        weights = weights / weights.sum()  # 정규화
        print(f"\nUsing weighted average with weights: {weights}")
    else:
        weights = None
        print(f"\nUsing simple average (equal weights)")
    
    # 클래스 이름 가져오기
    ind2class = data_loaders[0].dataset.ind2class
    num_classes = len(ind2class)
    
    # 클래스별 threshold tensor 생성
    threshold_tensor = None
    if class_thresholds is not None:
        thresholds = []
        for c in range(num_classes):
            class_name = ind2class[c]
            thr = class_thresholds.get(class_name, args.thr)
            thresholds.append(thr)
        threshold_tensor = torch.tensor(thresholds, device=device, dtype=torch.float32).view(1, num_classes, 1, 1)
    
    # Dice score 저장용
    dices_per_class = {class_name: [] for class_name in CLASSES}
    
    # 모든 데이터로더의 iterator 생성
    loader_iters = [iter(loader) for loader in data_loaders]
    num_batches = len(val_loader)
    
    print("\nValidating ensemble results...")
    with torch.no_grad():
        with tqdm(total=num_batches, desc="Ensemble validation") as pbar:
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)  # (B, C, H, W)
                
                # 각 데이터로더에서 배치 가져오기
                batch_data = []
                model_image_names_list = []
                for loader_iter in loader_iters:
                    model_images, model_image_names = next(loader_iter)
                    batch_data.append(model_images.to(device))
                    model_image_names_list.append(model_image_names)
                
                # 디버깅: 첫 번째 배치에서 이미지 이름 및 크기 확인
                if batch_idx == 0:
                    val_image_names = [val_dataset.fnames[i] for i in range(min(len(val_dataset), args.batch_size))]
                    print(f"\n[Debug] Validation images shape: {images.shape}, labels shape: {labels.shape}")
                    print(f"[Debug] Validation images (first batch): {val_image_names[:3]}...")
                    # 각 모델별 이미지 크기 확인
                    for i, (model_images, model_names) in enumerate(zip(batch_data, model_image_names_list)):
                        print(f"[Debug] Model {i+1} images shape: {model_images.shape}, names: {model_names[:3]}...")
                        if val_image_names[0] != model_names[0]:
                            print(f"[Warning] Model {i+1} image name mismatch! Validation: {val_image_names[0]}, Model: {model_names[0]}")
                
                # 각 모델로 추론 (validate.py와 동일한 방식)
                model_outputs = []
                
                # Crop 모드일 때 처리
                if args.crop_mode and args.crop_model_idx is not None and args.crop_model_idx < len(models):
                    crop_model_idx = args.crop_model_idx
                    full_model_indices = [i for i in range(len(models)) if i != crop_model_idx]
                    
                    # Full 모델들로 먼저 예측 (wrist ROI 추출용)
                    full_outputs_list = []
                    for i in full_model_indices:
                        outputs = models[i](batch_data[i])
                        outputs = unwrap_for_infer(outputs)
                        outputs = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                        outputs = torch.sigmoid(outputs)
                        full_outputs_list.append(outputs)
                    
                    # Full 모델들의 평균으로 wrist ROI 추출
                    if len(full_outputs_list) > 0:
                        full_pred = torch.mean(torch.stack(full_outputs_list), dim=0)  # (B, C, H, W)
                    else:
                        full_pred = None
                    
                    # 배치 내 각 이미지에 대해 crop 처리
                    batch_size = batch_data[0].shape[0]
                    crop_outputs_list = []
                    
                    for b in range(batch_size):
                        # 원본 이미지 읽기
                        image_name = model_image_names_list[0][b]  # 첫 번째 데이터로더의 이미지 이름 사용
                        image_path = osp.join(args.image_root, image_name)
                        ori_img = cv2.imread(image_path)
                        if ori_img is None:
                            # 이미지 읽기 실패 시 full 모델 예측 사용
                            crop_outputs_list.append(full_pred[b:b+1])
                            continue
                        
                        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
                        ori_h, ori_w = ori_img.shape[:2]
                        
                        # 예측 크기 (label 크기와 동일)
                        pred_h, pred_w = labels.shape[2], labels.shape[3]
                        
                        # Wrist ROI 좌표 추출 (예측 크기 기준)
                        ya_pred, yb_pred, xa_pred, xb_pred, found = get_wrist_roi_coords(
                            full_pred[b], pred_h, pred_w, margin_frac=0.15, threshold=0.5
                        )
                        
                        if found:
                            # 예측 크기 좌표를 원본 크기로 스케일링
                            scale_h = ori_h / pred_h
                            scale_w = ori_w / pred_w
                            ya = int(ya_pred * scale_h)
                            yb = int(yb_pred * scale_h)
                            xa = int(xa_pred * scale_w)
                            xb = int(xb_pred * scale_w)
                            
                            # 경계 체크
                            ya = max(0, min(ya, ori_h))
                            yb = max(0, min(yb, ori_h))
                            xa = max(0, min(xa, ori_w))
                            xb = max(0, min(xb, ori_w))
                            
                            # Crop
                            crop_img = ori_img[ya:yb, xa:xb]
                            
                            if crop_img.size > 0:
                                # Crop 이미지 전처리 (crop 모델의 resize 크기 사용)
                                crop_resize = data_loaders[crop_model_idx].dataset.transforms.transforms[0].height if hasattr(data_loaders[crop_model_idx].dataset.transforms, 'transforms') and len(data_loaders[crop_model_idx].dataset.transforms.transforms) > 0 else 1024
                                crop_tf = A.Compose([
                                    A.Resize(crop_resize, crop_resize),
                                    A.Normalize(),
                                ])
                                crop_transformed = crop_tf(image=crop_img)['image']
                                crop_tensor = torch.from_numpy(crop_transformed.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                                
                                # Crop 모델로 예측
                                crop_output = models[crop_model_idx](crop_tensor)
                                crop_output = unwrap_for_infer(crop_output)
                                # Label 크기로 interpolate
                                crop_output = F.interpolate(crop_output, size=(pred_h, pred_w), mode="bilinear", align_corners=False)
                                crop_output = torch.sigmoid(crop_output).squeeze(0)  # (C, H, W)
                                
                                # 예측 크기 좌표로 복원
                                crop_output_full = paste_crop_to_original(crop_output, pred_h, pred_w, (ya_pred, yb_pred, xa_pred, xb_pred))
                                crop_outputs_list.append(crop_output_full.unsqueeze(0))
                            else:
                                # Crop 실패 시 full 모델 예측 사용
                                crop_outputs_list.append(full_pred[b:b+1])
                        else:
                            # Wrist ROI를 찾지 못한 경우 full 모델 예측 사용
                            crop_outputs_list.append(full_pred[b:b+1])
                    
                    # Crop 모델 출력을 배치로 합치기
                    crop_outputs = torch.cat(crop_outputs_list, dim=0)  # (B, C, H, W)
                    model_outputs.append(crop_outputs)
                    
                    # Full 모델 출력도 추가
                    for i, full_out in enumerate(full_outputs_list):
                        model_outputs.insert(full_model_indices[i], full_out)
                else:
                    # 일반 모드: 모든 모델에 동일한 이미지 입력
                    for i, (model, model_images) in enumerate(zip(models, batch_data)):
                        # validate.py와 동일: 모델 직접 호출 -> unwrap_for_infer -> interpolate -> sigmoid
                        outputs = model(model_images)
                        
                        # 디버깅: 첫 번째 배치에서 각 모델 출력 크기 확인
                        if batch_idx == 0:
                            print(f"[Debug] Model {i+1} raw output shape: {outputs.shape}")
                        
                        # DeepSup 등 처리
                        outputs = unwrap_for_infer(outputs)
                        
                        # validate.py와 동일: label 크기로 interpolate
                        outputs = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                        
                        # sigmoid 적용
                        outputs = torch.sigmoid(outputs)
                        model_outputs.append(outputs)
                        
                        # 디버깅: 첫 번째 배치에서 최종 출력 크기 확인
                        if batch_idx == 0:
                            print(f"[Debug] Model {i+1} final output shape (after interpolate & sigmoid): {outputs.shape}")
                
                # 가중 평균으로 앙상블 (클래스별 선택적 앙상블 지원)
                if args.crop_model_idx is not None and args.crop_model_idx < len(models):
                    # 클래스별 선택적 앙상블: Wrist class는 crop 모델만, 나머지는 full 모델들만
                    ensemble_outputs = torch.zeros_like(model_outputs[0])
                    crop_model_idx = args.crop_model_idx
                    full_model_indices = [i for i in range(len(models)) if i != crop_model_idx]
                    
                    # Wrist class 인덱스 계산
                    wrist_indices = WRIST_INDICES
                    
                    if weights is not None:
                        # Full 모델들의 가중치 합 계산
                        full_weights_sum = sum(weights[i] for i in full_model_indices)
                        crop_weight = weights[crop_model_idx]
                        
                        # Full 모델들의 예측을 평균 (wrist ROI 추출용이었던 것)
                        if len(full_model_indices) > 0:
                            full_pred_avg = torch.mean(torch.stack([model_outputs[i] for i in full_model_indices]), dim=0)
                        else:
                            full_pred_avg = None
                        
                        for c in range(ensemble_outputs.shape[1]):  # 각 클래스별로
                            if c in wrist_indices:
                                # Wrist class: crop 모드일 때는 weighted ensemble, 아닐 때는 crop 모델만
                                if args.crop_mode and full_pred_avg is not None:
                                    # Crop 모드: full 모델과 crop 모델 weighted ensemble
                                    full_part = full_pred_avg[:, c, :, :]
                                    crop_part = model_outputs[crop_model_idx][:, c, :, :]
                                    # 가중치 정규화 (full + crop = 1.0)
                                    total_weight = full_weights_sum + crop_weight
                                    if total_weight > 0:
                                        w_full = full_weights_sum / total_weight
                                        w_crop = crop_weight / total_weight
                                        ensemble_outputs[:, c, :, :] = w_full * full_part + w_crop * crop_part
                                    else:
                                        ensemble_outputs[:, c, :, :] = (full_part + crop_part) / 2.0
                                else:
                                    # 일반 모드: crop 모델만 사용
                                    ensemble_outputs[:, c, :, :] = model_outputs[crop_model_idx][:, c, :, :]
                            else:
                                # 나머지 클래스: full 모델들만 앙상블 (가중치 정규화)
                                if full_weights_sum > 0:
                                    for i in full_model_indices:
                                        normalized_weight = weights[i] / full_weights_sum
                                        ensemble_outputs[:, c, :, :] += normalized_weight * model_outputs[i][:, c, :, :]
                                else:
                                    # 가중치가 0인 경우 단순 평균
                                    full_outputs = torch.stack([model_outputs[i][:, c, :, :] for i in full_model_indices])
                                    ensemble_outputs[:, c, :, :] = torch.mean(full_outputs, dim=0)
                    else:
                        # 단순 평균 버전
                        if len(full_model_indices) > 0:
                            full_pred_avg = torch.mean(torch.stack([model_outputs[i] for i in full_model_indices]), dim=0)
                        else:
                            full_pred_avg = None
                        
                        for c in range(ensemble_outputs.shape[1]):
                            if c in wrist_indices:
                                # Wrist class: crop 모드일 때는 weighted ensemble, 아닐 때는 crop 모델만
                                if args.crop_mode and full_pred_avg is not None:
                                    # Crop 모드: full 모델과 crop 모델 평균
                                    full_part = full_pred_avg[:, c, :, :]
                                    crop_part = model_outputs[crop_model_idx][:, c, :, :]
                                    ensemble_outputs[:, c, :, :] = (full_part + crop_part) / 2.0
                                else:
                                    # 일반 모드: crop 모델만 사용
                                    ensemble_outputs[:, c, :, :] = model_outputs[crop_model_idx][:, c, :, :]
                            else:
                                # 나머지 클래스: full 모델들만 평균
                                full_outputs = torch.stack([model_outputs[i][:, c, :, :] for i in full_model_indices])
                                ensemble_outputs[:, c, :, :] = torch.mean(full_outputs, dim=0)
                else:
                    # 기존 방식: 모든 클래스에 대해 일반 앙상블
                    if weights is not None:
                        ensemble_outputs = torch.zeros_like(model_outputs[0])
                        for i, output in enumerate(model_outputs):
                            ensemble_outputs += weights[i] * output
                    else:
                        ensemble_outputs = torch.mean(torch.stack(model_outputs), dim=0)
                
                # 디버깅: 첫 번째 배치에서 예측 분포 확인
                if batch_idx == 0:
                    print(f"[Debug] Ensemble output shape: {ensemble_outputs.shape}")
                    print(f"[Debug] Ensemble output range: [{ensemble_outputs.min():.4f}, {ensemble_outputs.max():.4f}]")
                    print(f"[Debug] Ensemble output mean: {ensemble_outputs.mean():.4f}")
                    print(f"[Debug] Ensemble output > 0.5 ratio: {(ensemble_outputs > 0.5).float().mean():.4f}")
                    print(f"[Debug] Label range: [{labels.min():.4f}, {labels.max():.4f}]")
                    print(f"[Debug] Label mean: {labels.mean():.4f}")
                    print(f"[Debug] Label > 0.5 ratio: {(labels > 0.5).float().mean():.4f}")
                
                # Threshold 적용
                if threshold_tensor is not None:
                    ensemble_outputs = (ensemble_outputs > threshold_tensor).float()
                else:
                    ensemble_outputs = (ensemble_outputs > args.thr).float()
                
                # 디버깅: 첫 번째 배치에서 threshold 적용 후 분포 확인
                if batch_idx == 0:
                    print(f"[Debug] After threshold, positive ratio: {ensemble_outputs.mean():.4f}")
                    print(f"[Debug] Label positive ratio: {labels.mean():.4f}")
                
                # 배치 내 각 샘플에 대해 Dice score 계산
                for i in range(ensemble_outputs.shape[0]):
                    pred = ensemble_outputs[i].detach().cpu().numpy()  # (C, H, W)
                    target = labels[i].detach().cpu().numpy()  # (C, H, W)
                    
                    # 디버깅: 첫 번째 배치의 첫 번째 샘플에 대해 상세 정보 출력
                    if batch_idx == 0 and i == 0:
                        val_img_name = val_dataset.fnames[batch_idx * args.batch_size + i] if batch_idx * args.batch_size + i < len(val_dataset) else "unknown"
                        print(f"\n[Debug] First sample: {val_img_name}")
                        print(f"[Debug] Pred shape: {pred.shape}, Target shape: {target.shape}")
                        print(f"[Debug] Pred sum per class (first 5): {pred[:5].sum(axis=(1,2))}")
                        print(f"[Debug] Target sum per class (first 5): {target[:5].sum(axis=(1,2))}")
                    
                    # 각 클래스별 Dice score 계산
                    for c in range(num_classes):
                        pred_mask = pred[c]  # (H, W)
                        target_mask = target[c]  # (H, W)
                        class_name = ind2class[c]
                        
                        dice = calculate_dice_score(pred_mask, target_mask)
                        dices_per_class[class_name].append(dice)
                    
                    # 디버깅: 첫 번째 배치의 첫 번째 샘플에 대해 평균 dice 출력
                    if batch_idx == 0 and i == 0:
                        first_sample_dices = [calculate_dice_score(pred[c], target[c]) for c in range(num_classes)]
                        print(f"[Debug] First sample avg dice: {np.mean(first_sample_dices):.4f}")
                        print(f"[Debug] First sample dice per class (first 5): {[f'{ind2class[c]}: {first_sample_dices[c]:.4f}' for c in range(min(5, num_classes))]}")
                
                pbar.update(1)
    
    # 클래스별 평균 Dice score 계산
    avg_dices_per_class = {class_name: np.mean(dices) for class_name, dices in dices_per_class.items()}
    
    # 전체 평균 Dice score
    avg_dice = np.mean(list(avg_dices_per_class.values()))
    
    return avg_dice, avg_dices_per_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", required=True, help="Paths to model files (.pt)")
    parser.add_argument("--weights", type=float, nargs="+", default=None, help="Weights for each model (if None, use equal weights)")
    parser.add_argument("--use_tta", action="store_true", help="Use TTA (Test Time Augmentation)")
    parser.add_argument("--image_root", type=str, default="/data/ephemeral/home/dataset/test/DCM")
    parser.add_argument("--thr", type=float, default=0.5, help="Default threshold for all classes")
    parser.add_argument("--thr_dict", type=str, default=None, help="JSON file path for class-specific thresholds")
    parser.add_argument("--output", type=str, default="./ensemble_output.csv", help="Output CSV path")
    parser.add_argument("--resize", type=int, nargs="+", default=[1024], help="Size(s) to resize images for each model (can specify multiple, one per model)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    parser.add_argument("--model_configs", type=str, default=None, help="JSON file path containing model configs for state_dict models. Format: [{\"model_name\": \"UnetPlusPlus\", \"encoder_name\": \"se_resnext101_32x4d\", ...}, ...]")
    parser.add_argument("--validate", action="store_true", help="Validate ensemble results against GT (no CSV output)")
    parser.add_argument("--label_root", type=str, default=None, help="Path to validation labels root (required for --validate)")
    parser.add_argument("--val_fold", type=int, default=0, help="Validation fold number (0-4, for --validate)")
    parser.add_argument("--log_file", type=str, default=None, help="Path to save validation log file (optional)")
    parser.add_argument("--crop_model_idx", type=int, default=None, help="Index of crop-trained model in models list (for wrist classes only). If None, use normal ensemble for all classes.")
    parser.add_argument("--crop_mode", action="store_true", help="Use crop mode: crop model receives cropped images instead of full images. Requires crop_model_idx to be set.")
    args = parser.parse_args()
    
    # model_configs JSON 파일 로드
    model_configs = None
    if args.model_configs:
        if not osp.exists(args.model_configs):
            raise FileNotFoundError(f"Model configs file not found: {args.model_configs}")
        with open(args.model_configs, 'r') as f:
            model_configs = json.load(f)
        if not isinstance(model_configs, list):
            raise ValueError("model_configs must be a list of dictionaries")
        if len(model_configs) != len(args.models):
            raise ValueError(f"Number of model configs ({len(model_configs)}) must match number of models ({len(args.models)})")
        print(f"Loaded model configs from: {args.model_configs}")
    args.model_configs = model_configs
    
    # 클래스별 threshold 로드
    class_thresholds = None
    if args.thr_dict:
        with open(args.thr_dict, 'r') as f:
            class_thresholds = json.load(f)
        print(f"Class-wise threshold loaded: {class_thresholds}")
    
    # resize 크기 확인 및 확장
    if len(args.resize) == 1:
        # 하나만 주어진 경우 모든 모델에 동일하게 적용
        resizes = args.resize * len(args.models)
        print(f"Using resize={args.resize[0]} for all {len(args.models)} models")
    elif len(args.resize) == len(args.models):
        # 모델 개수만큼 주어진 경우 각각 적용
        resizes = args.resize
        print(f"Using different resize sizes for each model: {resizes}")
    else:
        raise ValueError(f"Number of resize sizes ({len(args.resize)}) must be 1 or match number of models ({len(args.models)})")
    
    # Validation 모드인지 확인
    if args.validate:
        if args.label_root is None:
            raise ValueError("--label_root is required when using --validate")
        
        # Validation 모드: validate.py와 동일하게 image_root와 label_root에서 같은 데이터셋의 파일 가져오기
        # image_root는 train set의 이미지 경로를 가리켜야 함 (test set이 아님!)
        print(f"Creating validation data loader...")
        print(f"[Info] Using image_root: {args.image_root}")
        print(f"[Info] Using label_root: {args.label_root}")
        print(f"[Warning] In validation mode, image_root should point to TRAIN set images, not test set!")
        
        # validate.py와 동일한 방식으로 fnames와 labels 수집
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
        
        val_tf = A.Resize(height=resizes[0], width=resizes[0])  # 첫 번째 resize 사용
        
        # fnames와 labels를 numpy array로 변환 (XRayDataset의 GroupKFold가 numpy array를 기대함)
        fnames_array = np.array(fnames)
        labels_array = np.array(labels)
        
        val_dataset = XRayDataset(
            fnames=fnames_array,
            labels=labels_array,
            image_root=args.image_root,
            label_root=args.label_root,
            fold=args.val_fold,
            transforms=[val_tf],
            is_train=False
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=False
        )
        
        # Validation 데이터셋에서 사용하는 이미지 목록 가져오기 (fold 필터링된 것)
        val_fnames = val_dataset.fnames  # fold 필터링된 이미지 목록
        
        # 각 모델마다 다른 resize로 데이터로더 생성 (validation과 동일한 이미지 사용)
        data_loaders = []
        for i, resize in enumerate(resizes):
            print(f"Creating data loader {i+1}/{len(resizes)} with resize={resize}")
            tf = A.Resize(height=resize, width=resize)
            test_dataset = XRayInferenceDataset(val_fnames,  # validation과 동일한 이미지 사용
                                                args.image_root,
                                                transforms=tf)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
                drop_last=False
            )
            data_loaders.append(test_loader)
        
        # 앙상블 validation 수행
        avg_dice, dices_per_class = ensemble_validate(args, data_loaders, val_loader, class_thresholds)
        
        # 결과 출력
        result_lines = []
        result_lines.append("\n" + "="*50)
        result_lines.append("Ensemble Validation Results")
        result_lines.append("="*50)
        result_lines.append("\nClass-wise Dice Scores:")
        result_lines.append("-" * 50)
        
        for class_name in CLASSES:
            dice_score = dices_per_class[class_name]
            result_lines.append(f"{class_name:20s}: {dice_score:.6f}")
        
        result_lines.append("-" * 50)
        result_lines.append(f"Average Dice Score: {avg_dice:.6f}")
        result_lines.append("="*50)
        
        # 터미널에 출력
        for line in result_lines:
            print(line)
        
        # 로그 파일에 저장 (지정된 경우)
        if args.log_file:
            log_dir = osp.dirname(args.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            with open(args.log_file, 'w') as f:
                f.write('\n'.join(result_lines))
                f.write('\n')
            print(f"\nValidation log saved to: {args.log_file}")
    else:
        # 일반 추론 모드 (CSV 저장)
        # 이미지 파일 목록 수집
        fnames = sorted([
            osp.relpath(osp.join(root, fname), start=args.image_root)
            for root, _, files in os.walk(args.image_root)
            for fname in files
            if osp.splitext(fname)[1].lower() == ".png"
        ])
        
        # 각 모델마다 다른 resize로 데이터로더 생성
        data_loaders = []
        for i, resize in enumerate(resizes):
            print(f"Creating data loader {i+1}/{len(resizes)} with resize={resize}")
            tf = A.Resize(height=resize, width=resize)
            test_dataset = XRayInferenceDataset(fnames,
                                                args.image_root,
                                                transforms=tf)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
                drop_last=False
            )
            data_loaders.append(test_loader)
        
        # 앙상블 추론 수행
        rles, filename_and_class = ensemble_inference(args, data_loaders, class_thresholds)
        
        # CSV 저장
        classes, filename = zip(*[x.split("_", 1) for x in filename_and_class])
        
        image_name = [os.path.basename(f) for f in filename]
        
        df = pd.DataFrame({
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        })
        
        df.to_csv(args.output, index=False)
        print(f"\nEnsemble results saved to: {args.output}")

