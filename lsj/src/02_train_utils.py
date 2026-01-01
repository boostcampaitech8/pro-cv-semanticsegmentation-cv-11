def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, file_name='unet3+_RLROP_CL_AdamW_HF_251227.pt'):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def resize_to(x, ref):
    return x if x.shape[-2:] == ref.shape[-2:] else F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def validation(epoch, model, loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    set_seed()
    model.eval()
    dices = []

    with torch.no_grad():
        total_loss = 0

        # for images, masks in loader:
        for _, (images, masks) in tqdm(enumerate(loader), total=len(loader)):
            # images = images.cuda(non_blocking=True)
            # masks  = masks.cuda(non_blocking=True).float()  # BCE류/ Dice류 모두 float 필요
            images, masks = images.cuda(), masks.cuda().float()

            logits = model(images)  # SMP는 tensor logits 반환
            if isinstance(logits, (tuple, list)):
                d1, d2, d3, d4, d5 = logits
                d1 = resize_to(d1, masks); d2 = resize_to(d2, masks); d3 = resize_to(d3, masks); d4 = resize_to(d4, masks); d5 = resize_to(d5, masks)

                loss = (1.0*criterion(d1,masks) + 0.4*criterion(d2,masks) + 0.3*criterion(d3,masks) + 0.2*criterion(d4,masks) + 0.1*criterion(d5,masks))

                logits = d1 # 이거 안하면 오류남, logits이 그대로 tuple list로 받아져서
            else:
                # output_h, output_w = logits.size(-2), logits.size(-1)
                # mask_h, mask_w = masks.size(-2), masks.size(-1)

                # # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
                # if output_h != mask_h or output_w != mask_w:
                #     outputs = F.interpolate(logits, size=(mask_h, mask_w), mode="bilinear")
                logits = resize_to(logits, masks)  # ✅ resize를 logits에 반영
                loss = criterion(logits, masks)
            # total_loss += loss.item()
            total_loss += loss

            outputs = torch.sigmoid(logits)
            outputs = (outputs > thr).float().detach().cpu()
            masks = masks.detach().cpu().float()

            dice = dice_coef(outputs, masks)
            dices.append(dice)


    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    avg_dice = torch.mean(dices_per_class).item()

    return avg_dice

def unwrap_logits(model_out):
    # UNet_3Plus_DeepSup: (d1,d2,d3,d4,d5) 튜플
    return model_out[0] if isinstance(model_out, (tuple, list)) else model_out


def train(model, data_loader, val_loader, criterion, optimizer, scheduler=None):
    print(f'Start training..')
    set_seed()
    
    best_dice = 0.
    model = model.cuda()
    
    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            images = images.cuda()
            
            masks = masks.cuda().float() #수정
            
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                d1, d2, d3, d4, d5 = outputs
                d1 = resize_to(d1, masks); d2 = resize_to(d2, masks); d3 = resize_to(d3, masks); d4 = resize_to(d4, masks); d5 = resize_to(d5, masks)

                loss = (1.0*criterion(d1,masks) + 0.4*criterion(d2,masks) + 0.3*criterion(d3,masks) + 0.2*criterion(d4,masks) + 0.1*criterion(d5,masks))
            else:
                outputs = resize_to(outputs, masks)   # ✅ 추가 (혹시 크기 다르면 대비)
                loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                ## 완디비 train 설정 ##
                wandb.log({
                    "train/loss": loss.item(),
                    "epoch": epoch + 1,
                    "step": epoch * len(train_loader) + step
                })
             
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            if scheduler is not None:
                scheduler.step(dice)  # ✅ dice 기준으로 LR 조정

                # (선택) wandb에 현재 lr 기록
                try:
                    wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=epoch+1)
                except:
                    pass

            wandb.log({
                "val/mean_dice": dice,
                "epoch": epoch + 1
            })
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)
