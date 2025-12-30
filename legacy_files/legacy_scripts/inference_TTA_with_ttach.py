import os
import torch
import argparse
import json
import numpy as np
import pandas as pd
import os.path as osp
import albumentations as A
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import XRayInferenceDataset, CLASSES
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


def inference_with_tta(args, data_loader, class_thresholds=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model).to(device)
    model.eval()
    
    # 후처리를 포함한 모델 래퍼 생성
    model_wrapper = ModelWithPostProcess(model, target_size=(2048, 2048))
    model_wrapper.eval()
    
    # TTA 적용
    transforms = tta.Compose([
        HorizontalFlip(),
    ])
    tta_model = tta.SegmentationTTAWrapper(
        model_wrapper,
        transforms,
        merge_mode='mean'  # 평균
    )
    
    # 클래스별 threshold tensor 생성 - 반복문 밖에서 한 번만 생성하고 매 배치마다 재사용함.
    threshold_tensor = None
    if class_thresholds is not None:
        num_classes = len(data_loader.dataset.ind2class)
        thresholds = []
        for c in range(num_classes):
            class_name = data_loader.dataset.ind2class[c]
            # thr 지정 안 되어 있는 class는 thr 기본 인자 값 사용함 - ex. 0.5
            thr = class_thresholds.get(class_name, args.thr)
            thresholds.append(thr)
        # (1, C, 1, 1) 형태로 만들어서 broadcasting 가능하게 함
        threshold_tensor = torch.tensor(thresholds, device=device, dtype=torch.float32).view(1, num_classes, 1, 1)
    
    rles = []
    filename_and_class = []
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="[Inference with TTA (ttach)...]", disable=False) as pbar:
            for images, image_names in data_loader:
                images = images.to(device)
                
                # TTA 적용 (ttach 기본: logit space에서 평균)
                outputs = tta_model(images)  # (B, C, 2048, 2048) - logit space
                
                # sigmoid 적용 (ttach 기본 방식: 평균 후 sigmoid)
                outputs = torch.sigmoid(outputs)
                
                # 클래스별 threshold 적용 시
                if threshold_tensor is not None:
                    # GPU에서 broadcasting으로 한 번에 처리(한 번에 안하고 하나씩 하면 속도 엄청 느려짐)
                    outputs = (outputs > threshold_tensor).detach().cpu().numpy()
                else:
                    # 기본: 모든 클래스에 동일한 threshold 적용
                    outputs = (outputs > args.thr).detach().cpu().numpy()
                
                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{data_loader.dataset.ind2class[c]}_{image_name}")
                
                pbar.update(1)
                    
    return rles, filename_and_class


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the model to use")
    parser.add_argument("--image_root", type=str, default="/data/ephemeral/home/data/test/DCM")
    parser.add_argument("--thr", type=float, default=0.5, help="Default threshold for all classes")
    parser.add_argument("--thr_dict", type=str, default=None, help="JSON file path for class-specific thresholds(ex. {\"Pisiform\": 0.3, \"Trapezoid\": 0.3})")
    parser.add_argument("--output", type=str, default="./output.csv")
    parser.add_argument("--resize", type=int, default=1024, help="Size to resize images (both width and height)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    args = parser.parse_args()
    
    # 클래스별 threshold 로드
    class_thresholds = None
    if args.thr_dict:
        with open(args.thr_dict, 'r') as f:
            class_thresholds = json.load(f)
        print(f"Class-wise threshold loaded: {class_thresholds}")

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

    rles, filename_and_class = inference_with_tta(args, test_loader, class_thresholds)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(args.output, index=False)


