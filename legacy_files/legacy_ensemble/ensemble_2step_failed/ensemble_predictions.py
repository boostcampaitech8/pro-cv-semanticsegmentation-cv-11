import os
import sys
import argparse
import json
import numpy as np
import pandas as pd

# 프로젝트 루트를 path에 추가
sys.path.append('/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11')
from dataset import CLASSES

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


def load_predictions(pred_dir, model_names):
    """
    저장된 확률값들을 로드
    """
    predictions = []
    image_names = None
    
    for model_name in model_names:
        pred_path = os.path.join(pred_dir, f"{model_name}_probs.npy")
        metadata_path = os.path.join(pred_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        
        # 확률값 로드
        pred = np.load(pred_path)  # (N, C, H, W)
        predictions.append(pred)
        
        # 메타데이터 로드 (첫 번째 모델의 image_names 사용)
        if image_names is None:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    image_names = metadata.get("image_names", None)
        
        print(f"Loaded {model_name}: shape {pred.shape}")
    
    return predictions, image_names


def ensemble_predictions(predictions, weights=None):
    """
    여러 모델의 예측을 앙상블
    """
    if weights is None:
        # 평균
        ensemble_pred = np.mean(predictions, axis=0)
    else:
        # 가중 평균
        weights = np.array(weights)
        weights = weights / weights.sum()  # 정규화
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
    
    return ensemble_pred


def apply_thresholds(probs, class_thresholds, default_thr=0.5):
    """
    클래스별 threshold 적용
    """
    num_classes = probs.shape[1]
    thresholded = np.zeros_like(probs, dtype=np.uint8)
    
    for c in range(num_classes):
        class_name = CLASSES[c]
        thr = class_thresholds.get(class_name, default_thr)
        thresholded[:, c, :, :] = (probs[:, c, :, :] > thr).astype(np.uint8)
    
    return thresholded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, default="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/essemble_preds", help="Directory containing saved predictions")
    parser.add_argument("--model_names", type=str, nargs="+", required=True, help="Model names to ensemble (e.g., hrnet_w18_epoch28 hrnet_w48_epoch30)")
    parser.add_argument("--weights", type=float, nargs="+", default=None, help="Weights for each model (if None, use equal weights)")
    parser.add_argument("--thr", type=float, default=0.5, help="Default threshold for all classes")
    parser.add_argument("--thr_dict", type=str, default=None, help="JSON file path for class-specific thresholds")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (if None, auto-generate)")
    parser.add_argument("--output_dir", type=str, default="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/essemble", help="Directory to save ensemble results")
    args = parser.parse_args()
    
    # 가중치 검증
    if args.weights is not None:
        if len(args.weights) != len(args.model_names):
            raise ValueError(f"Number of weights ({len(args.weights)}) must match number of models ({len(args.model_names)})")
    
    # 클래스별 threshold 로드
    class_thresholds = None
    if args.thr_dict:
        with open(args.thr_dict, 'r') as f:
            class_thresholds = json.load(f)
        print(f"Class-wise threshold loaded: {class_thresholds}")
    
    # 저장된 예측값 로드
    print(f"Loading predictions from: {args.pred_dir}")
    predictions, image_names = load_predictions(args.pred_dir, args.model_names)
    
    # 앙상블
    print("Ensembling predictions...")
    ensemble_probs = ensemble_predictions(predictions, weights=args.weights)
    print(f"Ensemble shape: {ensemble_probs.shape}")
    
    # Threshold 적용
    if class_thresholds is not None:
        thresholded = apply_thresholds(ensemble_probs, class_thresholds, default_thr=args.thr)
    else:
        thresholded = (ensemble_probs > args.thr).astype(np.uint8)
    
    # RLE 인코딩
    rles = []
    filename_and_class = []
    
    for i, image_name in enumerate(image_names):
        for c, class_name in enumerate(CLASSES):
            segm = thresholded[i, c, :, :]
            rle = encode_mask_to_rle(segm)
            rles.append(rle)
            filename_and_class.append(f"{class_name}_{image_name}")
    
    # CSV 생성
    classes, filename = zip(*[x.split("_", 1) for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    # 출력 경로 생성
    if args.output is None:
        model_names_str = "_".join(args.model_names)
        weights_str = ""
        if args.weights:
            weights_str = "_weights_" + "_".join([f"{w:.2f}" for w in args.weights])
        thr_str = f"_thr{args.thr}" if not args.thr_dict else "_thrdict"
        output_filename = f"ensemble_{model_names_str}{weights_str}{thr_str}.csv"
        args.output = os.path.join(args.output_dir, output_filename)
    
    df.to_csv(args.output, index=False)
    print(f"Ensemble CSV saved to: {args.output}")

