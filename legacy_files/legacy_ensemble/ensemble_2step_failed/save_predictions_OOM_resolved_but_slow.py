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

# 프로젝트 루트를 path에 추가
sys.path.append('/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11')
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


def inference_without_threshold(args, data_loader, use_tta=False, save_path=None):
    """
    확률값만 반환 (threshold 적용 전)
    메모리 효율을 위해 배치 단위로 파일에 저장
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
    
    all_image_names = []
    num_samples = 0
    num_classes = None
    image_size = None
    
    # 첫 번째 배치로 shape 확인
    first_batch = True
    total_samples = len(data_loader.dataset)
    
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
                
                batch_size = outputs.shape[0]
                
                # 첫 번째 배치에서 shape 확인 및 메모리 매핑 파일 생성
                if first_batch:
                    num_classes = outputs.shape[1]
                    image_size = (outputs.shape[2], outputs.shape[3])
                    
                    # 메모리 매핑 파일 생성 (정확한 크기로 할당)
                    memmap_array = np.memmap(
                        save_path,
                        dtype=np.float32,
                        mode='w+',
                        shape=(total_samples, num_classes, image_size[0], image_size[1])
                    )
                    first_batch = False
                
                # 메모리 매핑 파일에 배치 단위로 저장
                start_idx = num_samples
                end_idx = num_samples + batch_size
                memmap_array[start_idx:end_idx] = outputs
                
                num_samples += batch_size
                all_image_names.extend(image_names)
                
                pbar.update(1)
    
    # 메모리 매핑 파일 닫기
    del memmap_array
    
    return num_samples, all_image_names, num_classes, image_size


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
    
    # 파일명 생성
    suffix = "_tta" if args.use_tta else ""
    save_path = osp.join(args.save_dir, f"{model_name}{suffix}_probs.npy")
    metadata_path = osp.join(args.save_dir, f"{model_name}{suffix}_metadata.json")
    
    # 추론 수행 (메모리 효율적으로 배치 단위로 저장)
    num_samples, image_names, num_classes, image_size = inference_without_threshold(
        args, test_loader, use_tta=args.use_tta, save_path=save_path
    )
    
    print(f"Predictions saved to: {save_path}")
    print(f"Shape: ({num_samples}, {num_classes}, {image_size[0]}, {image_size[1]})")
    
    # 메타데이터 저장
    metadata = {
        "model_name": model_name,
        "model_path": args.model,
        "use_tta": args.use_tta,
        "image_names": image_names,
        "num_images": len(image_names),
        "num_classes": num_classes,
        "image_size": image_size,
        "resize": args.resize,
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

