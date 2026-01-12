import os
import os.path as osp
import re, shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from dotenv import load_dotenv

from dataset import XRayDataset

# Wandb logging과 artifact를 활용하여 Checkpoint 업로드 기능
load_dotenv()  # ← 이 줄이 핵심

WANDB_API_KEY = os.getenv('WANDB_API_KEY')
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
wandb.login()

def format_lr_for_run_name(lr):
    """LR을 run name 형식으로 변환 (3e-3 -> 3e-3, 0.001 -> 1e-3)"""
    if lr >= 1:
        return str(int(lr))
    elif lr >= 0.1:
        return f"{lr:.1f}".replace('0.', '')
    else:
        # 과학적 표기법으로 변환
        lr_str = f"{lr:.0e}"
        # e-0X -> e-X 형식으로 변환
        lr_str = lr_str.replace('e-0', 'e-').replace('e+0', 'e+')
        return lr_str

def format_scheduler_name(sched_name):
    """Scheduler 이름을 짧은 형식으로 변환"""
    sched_lower = sched_name.lower()
    if 'cosineannealingwarmuprestarts' in sched_lower or 'cosinewarmup' in sched_lower:
        return 'coswarmup'
    elif 'cosineannealinglr' in sched_lower:
        return 'cos'
    elif 'reducelronplateau' in sched_lower:
        return 'plateau'
    else:
        return sched_lower[:8]

def format_loss_name(loss_name, loss_parameter=None):
    """Loss 이름을 run name 형식으로 변환"""
    loss_lower = loss_name.lower()
    if loss_lower == 'mixed':
        if loss_parameter and 'losses' in loss_parameter:
            loss_parts = []
            for loss in loss_parameter['losses']:
                loss_type = loss.get('name', '').lower()
                if 'bce' in loss_type:
                    loss_parts.append('bce')
                elif 'dice' in loss_type:
                    loss_parts.append('dice')
                elif 'focal' in loss_type:
                    loss_parts.append('focal')
                elif 'tversky' in loss_type:
                    loss_parts.append('tversky')
            return '+'.join(loss_parts) if loss_parts else 'mixed'
    return loss_lower[:8]

def generate_run_name(configs):
    """
    자동으로 run name 생성
    형식: {decoder}-{backbone}-{res}_{opt+lr}_{sched}_{loss}_e{epoch}_b{bs}_{expID}
    예: fcnhead-hrnet_w48-1024_adamw-1e-3_coswarmup_bce+dice_e30_b4_v1
    """
    # Decoder와 Backbone은 yaml에서 명시적으로 받음
    decoder = configs.get('run_name_decoder', 'unknown').lower()
    backbone = configs.get('run_name_backbone', 'unknown').lower()
    
    # Resolution
    res = str(configs.get('image_size', 1024))
    
    # Optimizer + LR
    opt = configs.get('optimizer_name', 'adamw').lower()
    lr = configs.get('lr', 1e-3)
    lr_str = format_lr_for_run_name(lr)
    opt_lr = f"{opt}-{lr_str}"
    
    # Scheduler
    sched = format_scheduler_name(configs.get('scheduler_name', 'unknown'))
    
    # Loss
    loss = format_loss_name(
        configs.get('loss_name', 'unknown'),
        configs.get('loss_parameter', None)
    )
    
    # Epoch
    epoch = configs.get('max_epoch', 30)
    
    # Batch size
    bs = configs.get('train_batch_size', 2)
    
    # Experiment ID
    exp_id = configs.get('run_name_exp_id', configs.get('experiment_id', 'v1'))
    
    # 조합
    run_name = f"{decoder}-{backbone}-{res}_{opt_lr}_{sched}_{loss}_e{epoch}_b{bs}_{exp_id}"
    
    return run_name

def set_wandb(configs):
    # Run name 자동 생성 여부 확인
    auto_generate = configs.get('auto_run_name', False)
    
    if auto_generate:
        run_name = generate_run_name(configs)
        print(f"[Wandb] Auto-generated run name: {run_name}")
    elif configs.get('experiment_detail'):
        run_name = configs['experiment_detail']
    else:
        run_name = 'default'
    
    wandb_run = wandb.init(
        entity=configs['team_name'],
        project=configs['project_name'],
        name=run_name,
        config={
                'model': configs['model_name'],
                'resize': configs['image_size'],
                'batch_size': configs['train_batch_size'],
                'loss_name': configs['loss_name'],
                'scheduler_name': configs['scheduler_name'],
                'learning_rate': configs['lr'],
                'epoch': configs['max_epoch'],
                "optimizer_name": configs['optimizer_name'],
            }
    )

    return wandb_run


