import os
import sys
import os.path as osp

# 프로젝트 루트를 path에 추가 (import 전에 실행)
sys.path.append('/data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11')

import torch
import argparse
import json
import time
import numpy as np
import pandas as pd
import albumentations as A
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import XRayInferenceDataset
import ttach as tta
from ttach import HorizontalFlip

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


def load_model(model_path, device, use_tta=False):
    """
    모델을 로드하고 TTA 래퍼를 생성
    
    Returns:
        tta_model: 추론에 사용할 모델 (TTA 적용 또는 일반)
    """
    model = torch.load(model_path).to(device)
    model.eval()
    
    # 후처리를 포함한 모델 래퍼 생성
    model_wrapper = ModelWithPostProcess(model, target_size=(2048, 2048))
    model_wrapper.eval()
    
    # TTA 적용 (ttach 기반)
    if use_tta:
        transforms = tta.Compose([
            HorizontalFlip(),
        ])
        tta_model = tta.SegmentationTTAWrapper(
            model_wrapper,
            transforms,
            merge_mode='mean'  # 평균
        )
    else:
        tta_model = model_wrapper
    
    return tta_model


def ensemble_inference(args, data_loader, class_thresholds=None):
    """
    여러 모델을 배치 단위로 앙상블하여 추론 수행 (메모리 효율적)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 모든 모델 로드
    print(f"Loading {len(args.models)} models...")
    models = []
    for i, model_path in enumerate(args.models):
        start_time = time.time()
        print(f"[{i+1}/{len(args.models)}] Loading: {osp.basename(model_path)}")
        tta_model = load_model(model_path, device, use_tta=args.use_tta)
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
    
    # 클래스 이름 가져오기 (dataset에서)
    ind2class = data_loader.dataset.ind2class
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
    print("\nProcessing batches (ensemble on-the-fly)...")
    rles = []
    filename_and_class = []
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Ensemble inference") as pbar:
            for images, image_names in data_loader:
                images = images.to(device)
                
                # 각 모델로 추론
                model_outputs = []
                for model in models:
                    outputs = model(images)  # (B, C, 2048, 2048) - logit space
                    outputs = torch.sigmoid(outputs)  # sigmoid 적용
                    model_outputs.append(outputs)
                
                # 가중 평균으로 앙상블
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", required=True, help="Paths to model files (.pt)")
    parser.add_argument("--weights", type=float, nargs="+", default=None, help="Weights for each model (if None, use equal weights)")
    parser.add_argument("--use_tta", action="store_true", help="Use TTA (Test Time Augmentation)")
    parser.add_argument("--image_root", type=str, default="/data/ephemeral/home/dataset/test/DCM")
    parser.add_argument("--thr", type=float, default=0.5, help="Default threshold for all classes")
    parser.add_argument("--thr_dict", type=str, default=None, help="JSON file path for class-specific thresholds")
    parser.add_argument("--output", type=str, default="./ensemble_output.csv", help="Output CSV path")
    parser.add_argument("--resize", type=int, default=1024, help="Size to resize images (both width and height)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    args = parser.parse_args()
    
    # 클래스별 threshold 로드
    class_thresholds = None
    if args.thr_dict:
        with open(args.thr_dict, 'r') as f:
            class_thresholds = json.load(f)
        print(f"Class-wise threshold loaded: {class_thresholds}")
    
    # 이미지 파일 목록 생성
    fnames = {
        osp.relpath(osp.join(root, fname), start=args.image_root)
        for root, _, files in os.walk(args.image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }
    
    tf = A.Resize(height=args.resize, width=args.resize)
    
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
    
    # 앙상블 추론 수행
    rles, filename_and_class = ensemble_inference(args, test_loader, class_thresholds)
    
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

