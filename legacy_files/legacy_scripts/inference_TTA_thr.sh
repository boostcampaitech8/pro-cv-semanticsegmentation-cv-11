#!/bin/bash

# 출력 디렉토리 생성
mkdir -p /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/outputs/HRNet_W18

python /data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/inference_TTA.py \
    /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W18/251230/hrnet_w18_aug_elastic_3_stage-best_28epoch_0.9720.pt \
    --image_root /data/ephemeral/home/dataset/test/DCM \
    --thr 0.5 \
    --thr_dict /data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/scripts/basic_runners/class_thresholds_hard.json \
    --output /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/outputs/HRNet_W18/hrnet_w18_aug_elastic_3_stage-best_28epoch_0.9720_tta_classwise_thr.csv \
    --resize 1024 \
    --batch_size 1

