import os
import os.path as osp
import json


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

import torch
import random
import warnings
import numpy as np
import albumentations as A

import argparse

from tqdm.auto import tqdm
from trainer import Trainer
from dataset import XRayDataset, WristCropDataset
from omegaconf import OmegaConf
from utils.wandb import set_wandb
# from utils.normalization import replace_bn_with_gn, count_bn_layers
from torch.utils.data import DataLoader, ConcatDataset
from loss.loss_mixer import LossMixer
from scheduler.scheduler_picker import SchedulerPicker
from optimizers.optimizer_picker import OptimizerPicker 
from models.model_picker import ModelPicker

warnings.filterwarnings('ignore')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def setup(cfg):
    # 이미지 파일명
    fnames = {
        osp.relpath(osp.join(root, fname), start=cfg.image_root)
        for root, _, files in os.walk(cfg.image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }

    # label json 파일명
    labels = {
        osp.relpath(osp.join(root, fname), start=cfg.label_root)
        for root, _, files in os.walk(cfg.label_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".json"
    }

    return np.array(sorted(fnames)), np.array(sorted(labels))


def main(cfg):
    wandb_run = set_wandb(cfg)
    set_seed(cfg.seed)

    fnames, labels = setup(cfg)
    
    train_transforms = [getattr(A, aug)(**params) 
                                         for aug, params in cfg.train_transform.items()]  
    
    val_transform = [getattr(A, aug)(**params) 
                                         for aug, params in cfg.val_transform.items()]

    # Train dataset (원본)
    train_dataset_full = XRayDataset(fnames,
                                     labels,
                                     cfg.image_root,
                                     cfg.label_root,
                                     fold=cfg.val_fold,
                                     transforms=train_transforms,
                                     is_train=True)
    
    # Wrist crop 옵션 확인
    wrist_crop_config = cfg.get('wrist_crop', {})
    if wrist_crop_config.get('enabled', False):
        train_dataset_wrist = WristCropDataset(fnames,
                                              labels,
                                              cfg.image_root,
                                              cfg.label_root,
                                              fold=cfg.val_fold,
                                              transforms=train_transforms,
                                              is_train=True,
                                              min_size=wrist_crop_config.get('min_size', 128),
                                              margin_frac=wrist_crop_config.get('margin_frac', 0.15))
        
        # mode에 따라 데이터셋 선택: "concat" (기본값) 또는 "crop_only"
        crop_mode = wrist_crop_config.get('mode', 'concat')
        if crop_mode == 'crop_only':
            train_dataset = train_dataset_wrist
            print(f"[Wrist Crop] Mode: crop_only - min_size: {wrist_crop_config.get('min_size', 128)}, "
                  f"margin_frac: {wrist_crop_config.get('margin_frac', 0.15)}")
            print(f"[Wrist Crop] Train dataset size: {len(train_dataset_wrist)} (crop only)")
        else:  # mode == 'concat' (기본값)
            train_dataset = ConcatDataset([train_dataset_full, train_dataset_wrist])
            print(f"[Wrist Crop] Mode: concat - min_size: {wrist_crop_config.get('min_size', 128)}, "
                  f"margin_frac: {wrist_crop_config.get('margin_frac', 0.15)}")
            print(f"[Wrist Crop] Train dataset size: {len(train_dataset_full)} (full) + {len(train_dataset_wrist)} (crop) = {len(train_dataset)}")
    else:
        train_dataset = train_dataset_full
    
    # Valid dataset (wrist_crop이 enabled이고 crop_only 모드면 동일하게 crop 적용)
    if wrist_crop_config.get('enabled', False) and wrist_crop_config.get('mode') == 'crop_only':
        valid_dataset = WristCropDataset(fnames,
                                        labels,
                                        cfg.image_root,
                                        cfg.label_root,
                                        fold=cfg.val_fold,
                                        transforms=val_transform,
                                        is_train=False,
                                        min_size=wrist_crop_config.get('min_size', 128),
                                        margin_frac=wrist_crop_config.get('margin_frac', 0.15))
        print(f"[Wrist Crop] Validation dataset: crop_only mode (same as training)")
    else:
        valid_dataset = XRayDataset(fnames,
                                    labels,
                                    cfg.image_root,
                                    cfg.label_root,
                                    fold=cfg.val_fold,
                                    transforms=val_transform,
                                    is_train=False)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=8, # 8 / 이거 줄이면 속도 엄청나게 감소함 / 1로 하니까 약 4배 느려짐
        drop_last=True,

    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        persistent_workers=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model 선택
    model_selector = ModelPicker()
    model = model_selector.get_model(cfg.model_name, **cfg.model_parameter)

    # # BatchNorm → GroupNorm 교체 (yaml 설정에 따라)
    # replace_bn = cfg.get('replace_bn_with_gn', False)
    # if replace_bn:
    #     bn_count_before = count_bn_layers(model)
    #     if bn_count_before > 0:
    #         gn_num_groups = cfg.get('gn_num_groups', 32)
    #         print(f"[Normalization] Replacing BatchNorm with GroupNorm...")
    #         print(f"[Normalization] Found {bn_count_before} BatchNorm2d layers")
    #         print(f"[Normalization] GroupNorm groups: {gn_num_groups}")
    #         model = replace_bn_with_gn(model, num_groups=gn_num_groups)
    #         bn_count_after = count_bn_layers(model)
    #         if bn_count_after == 0:
    #             print(f"[Normalization] Successfully replaced all BatchNorm layers with GroupNorm")
    #         else:
    #             print(f"[Warning] {bn_count_after} BatchNorm layers remain (may be in nested modules)")
    #     else:
    #         print(f"[Normalization] No BatchNorm2d layers found in model (may use LayerNorm or other normalization)")

    # 체크포인트에서 가중치 로드
    checkpoint_path = cfg.get('resume_from', None)
    if checkpoint_path:
        if not osp.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"[Resume] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 전체 모델이 저장된 경우
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
            print("[Resume] Loaded entire model from checkpoint")
        # state_dict만 저장된 경우
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("[Resume] Loaded model state_dict from checkpoint")
        # state_dict가 직접 저장된 경우
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            print("[Resume] Loaded model state_dict from checkpoint")
        else:
            # 기타 경우: 전체 모델로 간주
            model = checkpoint
            print("[Resume] Loaded entire model from checkpoint (fallback)")
    
    model.to(device)

    # optimizer는 선택
    optimizer_picker = OptimizerPicker()
    optimizer = optimizer_picker.get_optimizer(
        cfg.optimizer_name, 
        params=model.parameters(), 
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    # CosineAnnealingWarmupRestarts의 경우 steps 자동 계산
    scheduler_parameter = dict(cfg.scheduler_parameter) if cfg.scheduler_parameter else {}
    if cfg.scheduler_name == "CosineAnnealingWarmupRestarts":
        steps_per_epoch = len(train_loader)
        
        # yaml에서 warmup_epochs와 cycle_epochs를 읽거나 기본값 사용
        warmup_epochs = scheduler_parameter.pop('warmup_epochs', 1)
        cycle_epochs = scheduler_parameter.pop('cycle_epochs', 5)
        
        # 자동 계산
        warmup_steps = steps_per_epoch * warmup_epochs
        first_cycle_steps = steps_per_epoch * (warmup_epochs + cycle_epochs)
        
        # 기존 값이 있으면 덮어쓰지 않고, 없으면 자동 계산값 사용
        if 'warmup_steps' not in scheduler_parameter:
            scheduler_parameter['warmup_steps'] = warmup_steps
        if 'first_cycle_steps' not in scheduler_parameter:
            scheduler_parameter['first_cycle_steps'] = first_cycle_steps
        
        print(f"[Scheduler Auto-calculation] steps_per_epoch: {steps_per_epoch}, "
              f"warmup_epochs: {warmup_epochs}, cycle_epochs: {cycle_epochs}")
        print(f"[Scheduler Auto-calculation] warmup_steps: {scheduler_parameter['warmup_steps']}, "
              f"first_cycle_steps: {scheduler_parameter['first_cycle_steps']}")
    
    # scheduler 선택
    scheduler_selector = SchedulerPicker(optimizer)
    scheduler = scheduler_selector.get_scheduler(cfg.scheduler_name, **scheduler_parameter)
    
    # class weights JSON 파일 로드 (선택적)
    class_weights = None
    class_weights_path = cfg.get('class_weights_path', None)
    if class_weights_path:
        if not osp.exists(class_weights_path):
            raise FileNotFoundError(f"Class weights file not found: {class_weights_path}")
        with open(class_weights_path, 'r') as f:
            class_weights = json.load(f)
        print(f"[Class Weights] Loaded from: {class_weights_path}")
        print(f"[Class Weights] Total classes: {len(class_weights)}")
        # 손목뼈 클래스들의 가중치 출력 (예시)
        wrist_classes = ['Trapezium', 'Trapezoid', 'Capitate', 'Hamate', 
                        'Scaphoid', 'Lunate', 'Triquetrum', 'Pisiform']
        wrist_weights = {k: v for k, v in class_weights.items() if k in wrist_classes}
        if wrist_weights:
            print(f"[Class Weights] Wrist bone weights: {wrist_weights}")
    
    # loss_parameter에 class_weights 추가
    loss_parameter = dict(cfg.loss_parameter) if cfg.loss_parameter else {}
    if class_weights is not None:
        # 각 loss에 class_weights 전달
        if 'losses' in loss_parameter:
            for loss_config in loss_parameter['losses']:
                if 'params' not in loss_config:
                    loss_config['params'] = {}
                loss_config['params']['class_weights'] = class_weights
        else:
            # 단일 loss인 경우
            if 'params' not in loss_parameter:
                loss_parameter['params'] = {}
            loss_parameter['params']['class_weights'] = class_weights
    
    # loss 선택
    loss_selector = LossMixer()
    criterion = loss_selector.get_loss(cfg.loss_name, **loss_parameter)

    # Loss switching 설정 (yaml에서 선택적으로 제공)
    loss_switch_config = cfg.get('loss_switch', None)

    trainer = Trainer(
        model=model,
        device=device,
        wandb_run=wandb_run,
        train_loader=train_loader,
        val_loader=valid_loader,
        threshold=cfg.threshold,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        max_epoch=cfg.max_epoch,
        save_dir=cfg.save_dir,
        val_interval=cfg.val_interval,
        checkpoint_name_format=cfg.get('checkpoint_name_format', None),
        loss_selector=loss_selector,
        loss_switch_config=loss_switch_config,
        accum_steps=cfg.get('accum_steps', 1)  # gradient accumulation steps (기본값: 1)
    )

    trainer.train()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_train.yaml")

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
        
    main(cfg)