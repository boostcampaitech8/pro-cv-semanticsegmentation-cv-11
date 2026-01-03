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
                 checkpoint_name_format: str = None,
                 loss_selector = None,
                 loss_switch_config: dict = None,
                 accum_steps: int = 1 
                ):
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
        
        # Loss switching 관련 설정
        self.loss_selector = loss_selector
        self.loss_switch_config = loss_switch_config
        self.loss_switched = False
        self.consecutive_satisfied_epochs = 0  # 연속 만족 에폭 카운터
        
        # 3-stage loss switching 관련 설정
        self.loss_stages_config = loss_switch_config.get('loss_stages', None) if loss_switch_config else None
        self.current_stage = 1  # 현재 stage (1, 2, 3)
        self.stage2_consecutive = 0  # Stage 2 전환을 위한 연속 만족 카운터
        self.stage3_consecutive = 0  # Stage 3 전환을 위한 연속 만족 카운터
        # self.use_amp = True
        self.use_amp = False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.accum_steps = accum_steps


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

        self.optimizer.zero_grad(set_to_none=True)
        
        # 에포크 시작 시 현재 LR 및 Stage 출력
        current_lr = self.optimizer.param_groups[0]['lr']
        stage_info = f"Stage {self.current_stage}" if self.loss_stages_config is not None else ""
        print(f"Training epoch {epoch} start - Current LR: {current_lr:.6f} {stage_info}")

        #FP16
        # scaler = torch.cuda.amp.GradScaler(enabled=True)

        with tqdm(total=len(self.train_loader), desc=f"[Training Epoch {epoch}]", disable=False) as pbar:
            for step, (images, masks) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                masks  = masks.to(self.device, non_blocking=True)

                # (중요) dtype 정리: BCE/Dice는 float mask가 안전
                masks = masks.float()
                # 만약 0/255라면 이거까지:
                masks = (masks > 0).float()

                if step == 0 and epoch == 1:
                    print("mask min/max:", masks.min().item(), masks.max().item())
                    print("mask unique:", torch.unique(masks)[:10])
                    
                    print("masks:", masks.shape)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    # print("output shape:", outputs.shape)

                # ✅ loss는 fp32로 계산 (AMP 불안정 해결 핵심)
                with torch.cuda.amp.autocast(enabled=False):
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

                # 🔥 accumulation 핵심
                loss = loss / self.accum_steps

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    # self.scaler.step(self.optimizer)
                    # self.scaler.update()
                else:
                    loss.backward()
                    # self.optimizer.step()

                if (step + 1) % self.accum_steps == 0 or (step + 1) == len(self.train_loader):
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)

                    # CosineAnnealingWarmupRestarts는 step 단위로 동작하므로 매 배치마다 step() 호출
                    if self.scheduler_name == "CosineAnnealingWarmupRestarts":
                        self.scheduler.step()



                # total_loss += loss.item()
                total_loss += loss.item() * self.accum_steps
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
            
        train_end = time.time() - train_start 
        current_lr = self.optimizer.param_groups[0]['lr']
        stage_info = f"|| Stage: {self.current_stage}" if self.loss_stages_config is not None else ""
        print("Epoch {}, Train Loss: {:.4f} || LR: {:.6f} {} || Elapsed time: {} || ETA: {}\n".format(
            epoch,
            total_loss / len(self.train_loader),
            current_lr,
            stage_info,
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
                        
                        ### 251230 jsw) 이거 하면 메모리 사용량 줄어들 수도 있다고 함. ### 
                        del d2, d3, d4, d5
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

            train_log = {
                "train/epoch" : epoch,
                "train/loss" : train_loss,
                "train/lr": self.optimizer.param_groups[0]['lr']
            }
            # 3-stage loss switching 사용 시 현재 stage 로깅
            if self.loss_stages_config is not None:
                train_log["train/loss_stage"] = self.current_stage
            wandb.log(train_log, step=epoch)

            # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
            if epoch % self.val_interval == 0:
                avg_dice, dices_per_class, val_loss = self.validation(epoch)
                val_log = {
                    "val/loss": val_loss,
                    "val/avg_dice": avg_dice,
                }
                # 3-stage loss switching 사용 시 현재 stage 로깅
                if self.loss_stages_config is not None:
                    val_log["val/loss_stage"] = self.current_stage
                
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
                
                # 3-stage loss switching 체크
                if (self.loss_stages_config is not None and 
                    self.loss_selector is not None):
                    
                    # Stage 1 -> Stage 2 전환 체크
                    if self.current_stage == 1:
                        stage2_config = self.loss_stages_config.get('stage2', None)
                        if stage2_config and stage2_config.get('enabled', False):
                            conditions = stage2_config.get('conditions', {})
                            global_threshold = conditions.get('global_dice_threshold', 0.85)
                            class_thresholds = conditions.get('class_dice_thresholds', {})
                            consecutive_epochs = conditions.get('consecutive_epochs', 3)
                            
                            # 조건 체크
                            conditions_met = True
                            if avg_dice < global_threshold:
                                conditions_met = False
                            
                            if conditions_met and class_thresholds:
                                for class_name, threshold in class_thresholds.items():
                                    if class_name not in dices_per_class:
                                        conditions_met = False
                                        break
                                    if dices_per_class[class_name] < threshold:
                                        conditions_met = False
                                        break
                            
                            if conditions_met:
                                self.stage2_consecutive += 1
                                print(f"Stage 1->2 condition met ({self.stage2_consecutive}/{consecutive_epochs} consecutive epochs)")
                                
                                if self.stage2_consecutive >= consecutive_epochs:
                                    stage2_loss = stage2_config.get('loss', {})
                                    new_loss_name = stage2_loss.get('loss_name', 'Mixed')
                                    new_loss_parameter = stage2_loss.get('loss_parameter', {})
                                    
                                    # Learning rate 조정 (lr_multiplier가 설정된 경우)
                                    lr_multiplier = stage2_config.get('lr_multiplier', 1.0)
                                    if lr_multiplier != 1.0:
                                        # CosineAnnealingWarmupRestarts의 경우 max_lr, min_lr, base_max_lr 조정
                                        # 주의) 아직 테스트는 안 했음.
                                        if self.scheduler_name == "CosineAnnealingWarmupRestarts":
                                            current_max_lr = self.scheduler.max_lr
                                            current_min_lr = self.scheduler.min_lr
                                            current_base_max_lr = self.scheduler.base_max_lr
                                            
                                            new_max_lr = current_max_lr * lr_multiplier
                                            new_min_lr = current_min_lr * lr_multiplier
                                            new_base_max_lr = current_base_max_lr * lr_multiplier
                                            
                                            self.scheduler.max_lr = new_max_lr
                                            self.scheduler.min_lr = new_min_lr
                                            self.scheduler.base_max_lr = new_base_max_lr
                                            
                                            print(f"CosineAnnealingWarmupRestarts LR adjusted: max_lr {current_max_lr:.6f} -> {new_max_lr:.6f}, min_lr {current_min_lr:.6f} -> {new_min_lr:.6f} (multiplier: {lr_multiplier})")
                                        else:
                                            # 다른 스케줄러 (ReduceLROnPlateau 등)의 경우 기존 방식
                                            current_lr = self.optimizer.param_groups[0]['lr']
                                            new_lr = current_lr * lr_multiplier
                                            for param_group in self.optimizer.param_groups:
                                                param_group['lr'] = new_lr
                                            print(f"Learning rate adjusted: {current_lr:.6f} -> {new_lr:.6f} (multiplier: {lr_multiplier})")
                                    
                                    print(f"Stage 1 -> Stage 2 switching triggered at epoch {epoch}")
                                    self.criterion = self.loss_selector.get_loss(new_loss_name, **new_loss_parameter)
                                    self.current_stage = 2
                                    self.stage2_consecutive = 0
                                    
                                    wandb.log({
                                        "val/loss_stage": 2,
                                        "val/loss_stage_switch_epoch": epoch
                                    }, step=epoch)
                            else:
                                if self.stage2_consecutive > 0:
                                    print(f"Stage 1->2 condition not met, resetting counter (was {self.stage2_consecutive}/{consecutive_epochs})")
                                self.stage2_consecutive = 0
                    
                    # Stage 2 -> Stage 3 전환 체크
                    elif self.current_stage == 2:
                        stage3_config = self.loss_stages_config.get('stage3', None)
                        if stage3_config and stage3_config.get('enabled', False):
                            conditions = stage3_config.get('conditions', {})
                            global_threshold = conditions.get('global_dice_threshold', 0.95)
                            class_thresholds = conditions.get('class_dice_thresholds', {})
                            consecutive_epochs = conditions.get('consecutive_epochs', 3)
                            
                            # 조건 체크
                            conditions_met = True
                            if avg_dice < global_threshold:
                                conditions_met = False
                            
                            if conditions_met and class_thresholds:
                                for class_name, threshold in class_thresholds.items():
                                    if class_name not in dices_per_class:
                                        conditions_met = False
                                        break
                                    if dices_per_class[class_name] < threshold:
                                        conditions_met = False
                                        break
                            
                            if conditions_met:
                                self.stage3_consecutive += 1
                                print(f"Stage 2->3 condition met ({self.stage3_consecutive}/{consecutive_epochs} consecutive epochs)")
                                
                                if self.stage3_consecutive >= consecutive_epochs:
                                    stage3_loss = stage3_config.get('loss', {})
                                    new_loss_name = stage3_loss.get('loss_name', 'Mixed')
                                    new_loss_parameter = stage3_loss.get('loss_parameter', {})
                                    
                                    # Learning rate 조정 (lr_multiplier가 설정된 경우)
                                    lr_multiplier = stage3_config.get('lr_multiplier', 1.0)
                                    if lr_multiplier != 1.0:
                                        # CosineAnnealingWarmupRestarts의 경우 max_lr, min_lr, base_max_lr 조정
                                        # 주의) 아직 테스트는 안 했음.
                                        if self.scheduler_name == "CosineAnnealingWarmupRestarts":
                                            current_max_lr = self.scheduler.max_lr
                                            current_min_lr = self.scheduler.min_lr
                                            current_base_max_lr = self.scheduler.base_max_lr
                                            
                                            new_max_lr = current_max_lr * lr_multiplier
                                            new_min_lr = current_min_lr * lr_multiplier
                                            new_base_max_lr = current_base_max_lr * lr_multiplier
                                            
                                            self.scheduler.max_lr = new_max_lr
                                            self.scheduler.min_lr = new_min_lr
                                            self.scheduler.base_max_lr = new_base_max_lr
                                            
                                            print(f"CosineAnnealingWarmupRestarts LR adjusted: max_lr {current_max_lr:.6f} -> {new_max_lr:.6f}, min_lr {current_min_lr:.6f} -> {new_min_lr:.6f} (multiplier: {lr_multiplier})")
                                        else:
                                            # 다른 스케줄러 (ReduceLROnPlateau 등)의 경우 기존 방식
                                            current_lr = self.optimizer.param_groups[0]['lr']
                                            new_lr = current_lr * lr_multiplier
                                            for param_group in self.optimizer.param_groups:
                                                param_group['lr'] = new_lr
                                            print(f"Learning rate adjusted: {current_lr:.6f} -> {new_lr:.6f} (multiplier: {lr_multiplier})")
                                    
                                    print(f"Stage 2 -> Stage 3 switching triggered at epoch {epoch}")
                                    self.criterion = self.loss_selector.get_loss(new_loss_name, **new_loss_parameter)
                                    self.current_stage = 3
                                    self.stage3_consecutive = 0
                                    
                                    wandb.log({
                                        "val/loss_stage": 3,
                                        "val/loss_stage_switch_epoch": epoch
                                    }, step=epoch)
                            else:
                                if self.stage3_consecutive > 0:
                                    print(f"Stage 2->3 condition not met, resetting counter (was {self.stage3_consecutive}/{consecutive_epochs})")
                                self.stage3_consecutive = 0
                
                # ============================================================
                # 레거시 코드: 기존 단일 loss switching 방식 (사용 안 함)
                # 현재는 3-stage loss switching (위의 loss_stages_config)만 사용 중
                # 다음 실행 시 문제 없으면 이 블록 전체를 삭제해도 됨
                # ============================================================
                # elif (self.loss_switch_config is not None and 
                #       self.loss_switch_config.get('enabled', False) and 
                #       not self.loss_switched and 
                #       self.loss_selector is not None):
                #     
                #     conditions = self.loss_switch_config.get('conditions', {})
                #     global_threshold = conditions.get('global_dice_threshold', None)
                #     class_thresholds = conditions.get('class_dice_thresholds', {})
                #     consecutive_epochs = conditions.get('consecutive_epochs', 3)
                #     
                #     # 조건 체크
                #     conditions_met = True
                #     
                #     # 1. Global dice threshold 체크
                #     if global_threshold is not None:
                #         if avg_dice < global_threshold:
                #             conditions_met = False
                #     
                #     # 2. Class dice thresholds 체크
                #     if conditions_met and class_thresholds:
                #         for class_name, threshold in class_thresholds.items():
                #             if class_name not in dices_per_class:
                #                 conditions_met = False
                #                 break
                #             if dices_per_class[class_name] < threshold:
                #                 conditions_met = False
                #                 break
                #     
                #     # 조건 만족 여부에 따라 처리
                #     if conditions_met:
                #         self.consecutive_satisfied_epochs += 1
                #         print(f"Loss switch condition met ({self.consecutive_satisfied_epochs}/{consecutive_epochs} consecutive epochs)")
                #         
                #         # 연속 만족 에폭 수가 충족되면 loss 전환
                #         if self.consecutive_satisfied_epochs >= consecutive_epochs:
                #             loss_after_switch = self.loss_switch_config.get('loss_after_switch', {})
                #             new_loss_name = loss_after_switch.get('loss_name', 'DiceLoss')
                #             new_loss_parameter = loss_after_switch.get('loss_parameter', {})
                #             
                #             print(f"Loss switching triggered at epoch {epoch}")
                #             print(f"Conditions: global_dice >= {global_threshold}, class thresholds: {class_thresholds}")
                #             print(f"Switching to: {new_loss_name}")
                #             
                #             # 새로운 criterion 생성
                #             self.criterion = self.loss_selector.get_loss(new_loss_name, **new_loss_parameter)
                #             self.loss_switched = True
                #             
                #             wandb.log({
                #                 "val/loss_switched": 1,
                #                 "val/loss_switch_epoch": epoch
                #             }, step=epoch)
                #     else:
                #         # 조건 불만족 시 카운터 리셋
                #         if self.consecutive_satisfied_epochs > 0:
                #             print(f"Loss switch condition not met, resetting counter (was {self.consecutive_satisfied_epochs}/{consecutive_epochs})")
                #         self.consecutive_satisfied_epochs = 0
                
                # scheduler가 ReduceLROnPlateau라면 validation과정에서 lr update
                if self.scheduler_name == "ReduceLROnPlateau":
                    self.scheduler.step(avg_dice)

            # scheduler가 ReduceLROnPlateau가 아니고 CosineAnnealingWarmupRestarts도 아닌 경우에만 매 Epoch 마다 Lr update
            # CosineAnnealingWarmupRestarts는 train_epoch에서 매 배치마다 step() 호출
            if self.scheduler_name != "ReduceLROnPlateau" and self.scheduler_name != "CosineAnnealingWarmupRestarts":
                self.scheduler.step()