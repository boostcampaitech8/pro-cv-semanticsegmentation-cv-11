import os
import sys
import torch
import argparse
import json
import numpy as np
import os.path as osp
import albumentations as A
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import ttach as tta
from ttach import HorizontalFlip


def format_bytes(bytes_size):
    """바이트를 읽기 쉬운 형식으로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

# 프로젝트 루트를 path에 추가
sys.path.append('/data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11')
from dataset import XRayInferenceDataset

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


def inference_without_threshold(args, data_loader, use_tta=False):
    """
    확률값만 반환 (threshold 적용 전)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model).to(device)
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
    
    all_probs = []
    all_image_names = []
    
    # 메모리 사용량 출력 주기 (10배치마다)
    print_interval = 10
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="[Saving predictions...]", disable=False) as pbar:
            for batch_idx, (images, image_names) in enumerate(data_loader):
                images = images.to(device)
                
                # TTA 또는 일반 추론
                outputs = tta_model(images)  # (B, C, 2048, 2048) - logit space
                
                # sigmoid 적용 (ttach 기본 방식: 평균 후 sigmoid)
                outputs = torch.sigmoid(outputs)
                
                # 확률값 저장 (threshold 적용 전)
                outputs = outputs.detach().cpu().numpy()  # (B, C, H, W)
                
                all_probs.append(outputs)
                all_image_names.extend(image_names)
                
                # 메모리 사용량 출력 (일정 배치마다)
                if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == len(data_loader):
                    # 현재 배치 크기
                    batch_size_bytes = outputs.nbytes
                    
                    # 누적된 모든 배치의 총 메모리 사용량
                    total_memory_bytes = sum(arr.nbytes for arr in all_probs)
                    
                    # 리스트 오버헤드 (대략적)
                    list_overhead = sys.getsizeof(all_probs) + sum(sys.getsizeof(arr) for arr in all_probs)
                    
                    # 총 메모리 사용량
                    total_with_overhead = total_memory_bytes + list_overhead
                    
                    # 누적된 이미지 수
                    num_accumulated = sum(arr.shape[0] for arr in all_probs)
                    
                    print(f"\n[Batch {batch_idx + 1}/{len(data_loader)}] "
                          f"Accumulated: {num_accumulated} images | "
                          f"Current batch: {format_bytes(batch_size_bytes)} | "
                          f"Total memory: {format_bytes(total_memory_bytes)} "
                          f"(+ overhead: {format_bytes(total_with_overhead)})")
                
                pbar.update(1)
    
    # 모든 배치를 합치기 전 메모리 사용량 출력
    total_before_concat = sum(arr.nbytes for arr in all_probs)
    print(f"\n[Before concatenation] Total memory: {format_bytes(total_before_concat)}")
    
    # 모든 배치를 합침
    all_probs = np.concatenate(all_probs, axis=0)  # (N, C, H, W)
    
    # 합친 후 메모리 사용량 출력
    total_after_concat = all_probs.nbytes
    print(f"[After concatenation] Total memory: {format_bytes(total_after_concat)}")
    print(f"[Memory efficiency] Overhead: {format_bytes(total_after_concat - total_before_concat)} "
          f"({((total_after_concat - total_before_concat) / total_before_concat * 100):.2f}%)")
    
    return all_probs, all_image_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the model to use")
    parser.add_argument("--image_root", type=str, default="/data/ephemeral/home/dataset/test/DCM")
    parser.add_argument("--save_dir", type=str, default="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/essemble_preds", help="Directory to save predictions")
    parser.add_argument("--resize", type=int, default=1024, help="Size to resize images (both width and height)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    parser.add_argument("--use_tta", action="store_true", help="Use TTA (Test Time Augmentation)")
    args = parser.parse_args()
    
    # 모델 이름 추출 (경로에서 파일명 추출 후 .pt 제거)
    model_name = osp.splitext(osp.basename(args.model))[0]
    
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
    
    # 추론 수행
    probs, image_names = inference_without_threshold(args, test_loader, use_tta=args.use_tta)
    
    # 파일명 생성
    suffix = "_tta" if args.use_tta else ""
    save_path = osp.join(args.save_dir, f"{model_name}{suffix}_probs.npy")
    metadata_path = osp.join(args.save_dir, f"{model_name}{suffix}_metadata.json")
    
    # 확률값 저장
    np.save(save_path, probs)
    print(f"Predictions saved to: {save_path}")
    print(f"Shape: {probs.shape} (N={probs.shape[0]}, C={probs.shape[1]}, H={probs.shape[2]}, W={probs.shape[3]})")
    
    # 메타데이터 저장
    metadata = {
        "model_name": model_name,
        "model_path": args.model,
        "use_tta": args.use_tta,
        "image_names": image_names,
        "num_images": len(image_names),
        "num_classes": probs.shape[1],
        "image_size": (probs.shape[2], probs.shape[3]),
        "resize": args.resize,
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

