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

def set_wandb(configs):
    wandb_run = wandb.init(
        entity=configs['team_name'],
        project=configs['project_name'],
        name=configs['experiment_detail'],
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


