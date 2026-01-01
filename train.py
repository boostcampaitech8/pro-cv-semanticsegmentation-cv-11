import os
import os.path as osp

import torch
import random
import warnings
import numpy as np
import albumentations as A

import argparse

from tqdm.auto import tqdm
from trainer import Trainer
from dataset import XRayDataset
from omegaconf import OmegaConf
from utils.wandb import set_wandb
from torch.utils.data import DataLoader
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

    train_dataset = XRayDataset(fnames,
                                labels,
                                cfg.image_root,
                                cfg.label_root,
                                fold=cfg.val_fold,
                                transforms=train_transforms,
                                is_train=True)
    
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
        num_workers=8,
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
    
    # loss 선택
    loss_selector = LossMixer()
    criterion = loss_selector.get_loss(cfg.loss_name, **cfg.loss_parameter)

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
        loss_switch_config=loss_switch_config
    )

    trainer.train()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_train.yaml")

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
        
    main(cfg)