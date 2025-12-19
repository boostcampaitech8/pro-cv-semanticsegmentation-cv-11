import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from torch.utils.data import DataLoader
import wandb
import datetime

from torch.optim.lr_scheduler import CosineAnnealingLR
from loss import build_loss
from datasets import XRayDataset
from models import build_model
from utils import (
    set_seed,
    dice_coef,
    collect_png_json_pairs,
    load_yaml,
    logits_to_preds,
    multiclass_dice_coef,
    multiclass_dice_per_class  
)

# ---------------- CLASSES (고정) ----------------
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
# ------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config yaml")
    return parser.parse_args()

def build_optimizer(cfg, model):
    opt_name = cfg["optimizer"]["name"].lower()
    lr = float(cfg["optimizer"]["lr"])
    wd = float(cfg["optimizer"].get("weight_decay", 0.0))

    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    raise ValueError(f"Unsupported optimizer: {opt_name}")

def build_scheduler(cfg, optimizer):
    if "scheduler" not in cfg:
        return None

    sched_name = cfg["scheduler"]["name"].lower()

    if sched_name == "cosine":
        min_lr = float(cfg["scheduler"].get("min_lr", 1e-6))
        T_max = int(cfg["train"]["epochs"])  # 전체 epoch 기준
        return CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=min_lr
        )

    else:
        raise ValueError(f"Unsupported scheduler: {sched_name}")


@torch.no_grad()
def validate(model, val_loader, criterion, cfg):
    model.eval()
    dices = []
    total_loss = 0.0

    num_classes = cfg["model"]["num_classes"]
    dices_per_class = []

    for images, masks in val_loader:
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        logits = model(images)["out"]

        if masks.dim() == 4:
            masks_ce = masks.argmax(dim=1)
        else:
            masks_ce = masks
        masks_ce = masks_ce.long()

        loss = criterion(logits, masks_ce)
        total_loss += loss.item()

        preds = logits_to_preds(logits, cfg)

        # 전체 Dice (scalar)
        dice = multiclass_dice_coef(preds, masks, num_classes)
        dices.append(dice.item())

        # 클래스별 Dice (C,)
        dice_batch = multiclass_dice_per_class(
            preds, masks, num_classes
        )
        dices_per_class.append(dice_batch.unsqueeze(0).cpu())

    # (N, C) → (C,)
    dices_per_class = torch.cat(dices_per_class, dim=0)
    dices_per_class = dices_per_class.mean(dim=0)

    print("\n".join(
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ))

    return sum(dices)/len(dices), total_loss/len(val_loader)




def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    # ---- cfg에서 값 가져오기 (여기서만!) ----
    IMAGE_ROOT = cfg["data"]["train_image_root"]
    LABEL_ROOT = cfg["data"]["train_label_root"]
    IMAGE_SIZE = int(cfg["data"]["image_size"])

    SEED = int(cfg["seed"])

    BATCH_SIZE = int(cfg["train"]["batch_size"])
    VAL_BATCH_SIZE = int(cfg["train"].get("val_batch_size", BATCH_SIZE))
    NUM_WORKERS = int(cfg["train"].get("num_workers", 4))
    EPOCHS = int(cfg["train"]["epochs"])
    VAL_INTERVAL = int(cfg["train"].get("val_interval", 1))
    THR = float(cfg["train"].get("threshold", 0.5))

    SAVE_DIR = cfg["checkpoint"]["save_dir"]
    CKPT_NAME = cfg["checkpoint"]["filename"]
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- seed/device ----
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- transforms ----
    tf = A.Resize(IMAGE_SIZE, IMAGE_SIZE)

    # ---- dataset ----
    pngs, jsons = collect_png_json_pairs(IMAGE_ROOT, LABEL_ROOT)

    train_ds = XRayDataset(
        image_root=IMAGE_ROOT,
        label_root=LABEL_ROOT,
        classes=CLASSES,
        pngs=pngs,
        jsons=jsons,
        is_train=True,
        transforms=tf
    )
    val_ds = XRayDataset(
        image_root=IMAGE_ROOT,
        label_root=LABEL_ROOT,
        classes=CLASSES,
        pngs=pngs,
        jsons=jsons,
        is_train=False,
        transforms=tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    # val은 메모리 이슈 줄이려고 workers 0 권장
    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    print("[DEBUG] #pngs:", len(pngs), " #jsons:", len(jsons), flush=True)
    print("[DEBUG] train_ds:", len(train_ds), " val_ds:", len(val_ds), flush=True)
    print("[DEBUG] train_loader:", len(train_loader), " val_loader:", len(val_loader), flush=True)


    # ---- model/loss/opt ----
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # ---- wandb ----
    if cfg["wandb"]["use"]:
        os.environ["WANDB_API_KEY"] = cfg["wandb"]["api_key"]
        wandb.login()
        wandb.init(
            entity=cfg["wandb"]["entity"],
            project=cfg["wandb"]["project"],
            name=cfg["wandb"]["name"],
            config=cfg,
        )

    best_dice = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for step, (images, masks) in enumerate(train_loader, start=1):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            logits = model(images)["out"]
            if masks.dim() == 4:
                masks_ce = masks.argmax(dim=1)
            else:
                masks_ce = masks
            masks_ce = masks_ce.long()


            loss = criterion(logits, masks_ce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(), 4)}',
                    flush=True
                )

                if cfg["wandb"]["use"]:
                    global_step = epoch* len(train_loader) + step
                    wandb.log({
                        "lr": optimizer.param_groups[0]["lr"],
                        "train/loss": loss.item(),
                        "epoch": epoch+1,
                        "step": global_step
                    })
        if scheduler is not None:
            scheduler.step()
        # ---- validation ----
        if (epoch+1) % VAL_INTERVAL == 0:
            val_dice, val_loss = validate(
                model,
                val_loader,
                criterion,
                cfg
            )

            if cfg["wandb"]["use"]:
                wandb.log({
                "val/mean_dice": val_dice,
                "epoch": epoch + 1
                })


            if val_dice > best_dice:
                best_dice = val_dice
                save_path = os.path.join(SAVE_DIR, CKPT_NAME)
                torch.save(model.state_dict(), save_path)
                print(f"[Best] epoch={epoch} mean_dice={best_dice:.4f} -> saved: {save_path}")

    if cfg["wandb"]["use"]:
        wandb.finish()


if __name__ == "__main__":
    main()
