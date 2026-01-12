# train.py
import os
import argparse
import datetime
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.optim import AdamW, Adam
import torch_optimizer as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from models import build_model
from dataset import collect_png_json_pairs, XRayDataset
from utils import (
    load_yaml, set_seed, build_loss,
    multilabel_dice_per_class, save_checkpoint,
    init_wandb
)

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="train.yaml")
    return p.parse_args()


def build_optimizer(cfg, model):
    ocfg = cfg["optimizer"]
    name = ocfg["name"].lower()
    lr = float(ocfg["lr"])
    wd = float(ocfg.get("weight_decay", 0.0))

    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamp":
        return optim.AdamP(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(cfg, optimizer):
    scfg = cfg.get("scheduler", None)
    if scfg is None:
        return None

    name = scfg["name"].lower()

    # -----------------------------------
    # 1. cosine + warmup + restarts (katsura-jp)
    # -----------------------------------
    if name in ["cosine_warmup", "cosine_warmup_restarts"]:
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=int(scfg["first_cycle_steps"]),
            cycle_mult=float(scfg.get("cycle_mult", 1.0)),
            max_lr=float(scfg["max_lr"]),
            min_lr=float(scfg.get("min_lr", 1e-6)),
            warmup_steps=int(scfg.get("warmup_steps", 0)),
            gamma=float(scfg.get("gamma", 1.0)),
        )

    # -----------------------------------
    # 2. plain cosine (epoch 단위)
    # -----------------------------------
    if name == "cosine":
        min_lr = float(scfg.get("min_lr", 1e-6))
        T_max = int(cfg["train"]["epochs"])
        return CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=min_lr
        )

    # -----------------------------------
    # 3. no scheduler
    # -----------------------------------
    if name in ["none", "noscheduler"]:
        return None

    raise ValueError(f"Unsupported scheduler: {name}")



def build_transforms(cfg):
    size = int(cfg["data"]["image_size"])

    # 노트북 스타일: 보존형 + 약한 대비/노이즈 정도만 추천
    train_tf = A.Compose([
        A.Resize(size, size),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=5, p=0.3),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.15),
        A.Normalize(),
        ToTensorV2(transpose_mask=True),
    ])

    valid_tf = A.Compose([
        A.Resize(size, size),
        A.Normalize(),
        ToTensorV2(transpose_mask=True),
    ])
    return train_tf, valid_tf


@torch.no_grad()
def validate(model, loader, criterion, cfg):
    model.eval()
    thr = float(cfg["train"].get("threshold", 0.5))
    total_loss = 0.0
    dices = []

    for images, masks in tqdm(loader, desc="valid", leave=False):
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        dice_per_class = multilabel_dice_per_class(probs, masks, thr=thr)  # (C,)
        dices.append(dice_per_class.unsqueeze(0).cpu())

    dices = torch.cat(dices, dim=0).mean(dim=0)  # (C,)
    mean_dice = dices.mean()

    # 출력
    print("\n[Validation | Dice per class]")
    for c, d in zip(CLASSES, dices):
        print(f"{c:<12}: {d.item():.4f}")
    print(f"[Validation] Mean Dice: {mean_dice.item():.4f}")

    return mean_dice.item(), total_loss / max(1, len(loader))


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(int(cfg.get("seed", 21)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    image_root = cfg["data"]["train_image_root"]
    label_root = cfg["data"]["train_label_root"]
    fold = int(cfg["data"].get("fold", 0))
    n_splits = int(cfg["data"].get("n_splits", 5))

    train_tf, valid_tf = build_transforms(cfg)

    pngs, jsons = collect_png_json_pairs(image_root, label_root)

    train_ds = XRayDataset(
        image_root=image_root,
        label_root=label_root,
        classes=CLASSES,
        pngs=pngs,
        jsons=jsons,
        fold=fold,
        n_splits=n_splits,
        is_train=True,
        transforms=train_tf
    )
    val_ds = XRayDataset(
        image_root=image_root,
        label_root=label_root,
        classes=CLASSES,
        pngs=pngs,
        jsons=jsons,
        fold=fold,
        n_splits=n_splits,
        is_train=False,
        transforms=valid_tf
    )

    tcfg = cfg["train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg.get("val_batch_size", tcfg["batch_size"])),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # model/loss/opt
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    use_amp = bool(cfg["train"].get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # wandb
    use_wandb, wcfg = init_wandb(cfg)
    if use_wandb:
        import wandb
        wandb.login(key=wcfg.get("api_key", None))
        wandb.init(
            entity=wcfg.get("entity", None),
            project=wcfg.get("project", "seg"),
            name=wcfg.get("name", f"run_{datetime.datetime.now().strftime('%m%d_%H%M')}"),
            config=cfg,
        )

    # checkpoint
    ckpt_dir = cfg["checkpoint"]["save_dir"]
    ckpt_name = cfg["checkpoint"]["filename"]
    best_path = os.path.join(ckpt_dir, ckpt_name)
    best_dice = -1.0

    epochs = int(tcfg["epochs"])
    val_interval = int(tcfg.get("val_interval", 1))

    for epoch in range(epochs):
        model.train()
        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)   # ✅ smp SegFormer
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if isinstance(scheduler, CosineAnnealingWarmupRestarts):
                scheduler.step()

            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{epochs}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(), 4)}',
                    flush=True
                )

                if cfg["wandb"]["use"]:
                    global_step = epoch * len(train_loader) + step
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1,
                            "step": global_step
                        },
                        step=global_step
                    )

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        if (epoch + 1) % val_interval == 0:
            val_dice, val_loss = validate(model, val_loader, criterion, cfg)

            if cfg["wandb"]["use"]:
                wandb.log({
                    "val/mean_dice": val_dice,
                    "val/loss": val_loss,
                    "epoch": epoch + 1,
                })

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), best_path)
                print(f"[Best] epoch={epoch+1} mean_dice={best_dice:.4f}")


    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
