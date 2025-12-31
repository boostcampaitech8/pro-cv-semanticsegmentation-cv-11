import os
import time
import wandb
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm
from datetime import timedelta
from torch.utils.data import DataLoader

# from train import set_seed

def resize_to(x, ref):
    return x if x.shape[-2:] == ref.shape[-2:] else F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

def dice_coef(y_true, y_pred):
        y_true_f = y_true.flatten(2)
        y_pred_f = y_pred.flatten(2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)
        eps = 0.0001
        return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

class Trainer:
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 wandb_run: wandb.run, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 threshold: float,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler,
                 criterion: torch.nn.modules.loss._Loss,
                 max_epoch: int,
                 save_dir: str,
                 val_interval: int,
                 checkpoint_name_format: str = None):
        self.model = model
        self.device = device
        self.wandb_run = wandb_run
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.optimzier_name = optimizer.__class__.__name__
        self.scheduler = scheduler
        self.scheduler_name = scheduler.__class__.__name__
        self.criterion = criterion
        self.max_epoch = max_epoch
        self.save_dir = save_dir
        self.threshold = threshold
        self.val_interval = val_interval
        self.checkpoint_name_format = checkpoint_name_format or "best_{epoch}epoch_{dice_score:.4f}.pt"
        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)


    def save_model(self, epoch, dice_score, before_path):
        # checkpoint 저장 폴더 생성
        if not osp.isdir(self.save_dir):
            # os.mkdir(self.save_dir) # mkdir은 상위 디렉토리가 없으면 안만들어짐
            os.makedirs(self.save_dir, exist_ok=True)

        if before_path != "" and osp.exists(before_path):
            os.remove(before_path)

        # config에서 지정한 이름 형식 사용
        checkpoint_name = self.checkpoint_name_format.format(
            epoch=epoch,
            dice_score=dice_score
        )
        output_path = osp.join(self.save_dir, checkpoint_name)
        torch.save(self.model, output_path)
        return output_path


    def train_epoch(self, epoch):
        train_start = time.time()
        self.model.train()
        total_loss = 0.0

        #FP16
        # scaler = torch.cuda.amp.GradScaler(enabled=True)

        with tqdm(total=len(self.train_loader), desc=f"[Training Epoch {epoch}]", disable=False) as pbar:
            for images, masks in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                masks  = masks.to(self.device, non_blocking=True)

                # (중요) dtype 정리: BCE/Dice는 float mask가 안전
                masks = masks.float()
                # 만약 0/255라면 이거까지:
                # masks = (masks > 0).float()

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(images)

                # ✅ loss는 fp32로 계산 (AMP 불안정 해결 핵심)
                if isinstance(outputs, (tuple, list)):
                    d1, d2, d3, d4, d5 = outputs
                    d1 = resize_to(d1, masks); d2 = resize_to(d2, masks); d3 = resize_to(d3, masks); d4 = resize_to(d4, masks); d5 = resize_to(d5, masks)

                    loss = (1.0*self.criterion(d1.float(), masks) +
                            0.4*self.criterion(d2.float(), masks) +
                            0.3*self.criterion(d3.float(), masks) +
                            0.2*self.criterion(d4.float(), masks) +
                            0.1*self.criterion(d5.float(), masks))
                else:
                    outputs = resize_to(outputs, masks)
                    loss = self.criterion(outputs.float(), masks)

                # (디버깅/안전) NaN/Inf 체크
                if not torch.isfinite(loss):
                    print("NaN/Inf loss!", loss.item())
                    print("outputs:", outputs.dtype, outputs.min().item(), outputs.max().item())
                    print("masks:", masks.dtype, masks.min().item(), masks.max().item())
                    raise RuntimeError("Invalid loss")

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()


                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
            
        train_end = time.time() - train_start 
        current_lr = self.optimizer.param_groups[0]['lr']
        print("Epoch {}, Train Loss: {:.4f} || LR: {:.6f} || Elapsed time: {} || ETA: {}\n".format(
            epoch,
            total_loss / len(self.train_loader),
            current_lr,
            timedelta(seconds=train_end),
            timedelta(seconds=train_end * (self.max_epoch - epoch))
        ))
        return total_loss / len(self.train_loader)
    

    def validation(self, epoch):
        val_start = time.time()
        # set_seed()
        self.model.eval()

        total_loss = 0
        dices = []

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f'[Validation Epoch {epoch}]', disable=False) as pbar:
                for images, masks in self.val_loader:
                    images, masks = images.to(self.device), masks.to(self.device).float()
                    outputs = self.model(images)
                    if isinstance(outputs, (tuple, list)):
                        d1, d2, d3, d4, d5 = outputs
                        d1 = resize_to(d1, masks); d2 = resize_to(d2, masks); d3 = resize_to(d3, masks); d4 = resize_to(d4, masks); d5 = resize_to(d5, masks)

                        loss = (1.0*self.criterion(d1,masks) + 0.4*self.criterion(d2,masks) + 0.3*self.criterion(d3,masks) + 0.2*self.criterion(d4,masks) + 0.1*self.criterion(d5,masks))

                        outputs = d1 # 이거 안하면 오류남, outputs이 그대로 tuple list로 받아져서
                    else:
                        outputs = resize_to(outputs, masks)  # ✅ resize를 outputs에 반영
                        loss = self.criterion(outputs, masks)

                    # output_h, output_w = outputs.size(-2), outputs.size(-1)
                    # mask_h, mask_w = masks.size(-2), masks.size(-1)

                    # # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
                    # if output_h != mask_h or output_w != mask_w:
                    #     outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                    
                    # loss = self.criterion(outputs, masks)
                    total_loss += loss.item()

                    outputs = torch.sigmoid(outputs)
                    ## Dice 계산과정을 gpu에서 진행하도록 변경
                    outputs = (outputs > self.threshold)
                    dice = dice_coef(outputs, masks)
                    dices.append(dice.detach().cpu())
                    pbar.update(1)
                    pbar.set_postfix(dice=torch.mean(dice).item(), loss=loss.item())

        val_end = time.time() - val_start
        dices = torch.cat(dices, 0)
        dices_per_class = torch.mean(dices, 0)
        dice_str = [
            f"{c:<12}: {d.item():.4f}"
            for c, d in zip(self.val_loader.dataset.class2ind, dices_per_class)
        ]

        dice_str = "\n".join(dice_str)
        print(dice_str)
        
        avg_dice = torch.mean(dices_per_class).item()
        print("avg_dice: {:.4f}".format(avg_dice))
        print("Validation Loss: {:.4f} || Elapsed time: {}\n".format(
            total_loss / len(self.val_loader),
            timedelta(seconds=val_end),
        ))

        # class_dice_dict = {f"{c}'s dice score" : d for c, d in zip(self.val_loader.dataset.class2ind, dices_per_class)}
        # 따로 변환 없이 원본 써도 괜찮을 것 같음(wandb logging도 깔끔)
        class_dice_dict = {c: d.item() for c, d in zip(self.val_loader.dataset.class2ind.keys(), dices_per_class)}
        
        return avg_dice, class_dice_dict, total_loss / len(self.val_loader)
    


    def train(self):
        print(f'Start training..')
        # set_seed()

        best_dice = 0.
        before_path = ""
        
        for epoch in range(1, self.max_epoch + 1):

            train_loss = self.train_epoch(epoch)

            wandb.log({
                "train/epoch" : epoch,
                "train/loss" : train_loss,
                "train/lr": self.optimizer.param_groups[0]['lr']
            }, step=epoch)

            # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
            if epoch % self.val_interval == 0:
                avg_dice, dices_per_class, val_loss = self.validation(epoch)
                val_log = {
                    "val/loss": val_loss,
                    "val/avg_dice": avg_dice,
                }
                # class별 dice를 val/{class_name}_dice 형식으로 추가
                for class_name, dice_score in dices_per_class.items():
                    # 클래스 이름을 wandb-friendly 형식으로 변환 (공백, 하이픈을 언더스코어로) -> 굳이 필요 없을 듯.
                    # class_key = class_name.lower().replace(" ", "_").replace("-", "_")
                    val_log[f"val/{class_name}_dice"] = dice_score
                
                wandb.log(val_log, step=epoch)
                
                if best_dice < avg_dice:
                    print(f"Best performance at epoch: {epoch}, {best_dice:.4f} -> {avg_dice:.4f}\n")
                    best_dice = avg_dice
                    before_path = self.save_model(epoch, best_dice, before_path)
                    
                # scheduler가 ReduceLROnPlateau라면 validation과정에서 lr update
                if self.scheduler_name == "ReduceLROnPlateau":
                    self.scheduler.step(avg_dice)

            # scheduler가 ReduceLROnPlateau가 아니라면 매 Epoch 마다 Lr update
            if self.scheduler_name != "ReduceLROnPlateau":
                self.scheduler.step()