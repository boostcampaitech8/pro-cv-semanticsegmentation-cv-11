#!/bin/bash

# 출력 디렉토리 생성
mkdir -p /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/outputs/HRNet_W18

python /data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/inference.py \
    /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W18/bce03_dice07-best_30epoch_0.9701.pt \
    --image_root /data/ephemeral/home/dataset/test/DCM \
    --thr 0.5 \
    --thr_dict /data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/configs/mmseg/class_thresholds.json \
    --output /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/outputs/HRNet_W18/hrnet_w18-bce03_dice07-classwise_thr_0.3.csv \
    --resize 1024