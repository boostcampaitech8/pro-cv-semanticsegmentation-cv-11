#!/bin/bash

# 출력 디렉토리 생성
mkdir -p /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/for_vis/HRNet_W18

python /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/inference.py \
    /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W18/bce03_dice07-best_30epoch_0.9701.pt \
    --image_root /data/ephemeral/home/dataset/fold0/val/DCM \
    --thr 0.5 \
    --output /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/for_vis/HRNet_W18/hrnet_w18-bce03_dice07-for_vis.csv \
    --resize 1024 \
    --batch_size 1