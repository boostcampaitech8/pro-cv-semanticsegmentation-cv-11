#!/bin/bash

# 개별 모델의 확률값을 저장하는 스크립트
# 사용 시 아래 인자들을 수정하여 사용하세요

# 저장 디렉토리 생성
mkdir -p /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/ensemble_preds

python /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/scripts/ensemble/save_predictions.py \
    /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W18/251230/hrnet_w18_aug_elastic_3_stage-best_28epoch_0.9720.pt \
    --image_root /data/ephemeral/home/dataset/test/DCM \
    --save_dir /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/ensemble_preds \
    --resize 1024 \
    --batch_size 1 \
    --use_tta

python /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/scripts/ensemble/save_predictions.py \
    /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W18/251228_unetpp_simple/hrnet_w18_unetpp_with_smp_elastic_aug-29epoch_0.9710.pt \
    --image_root /data/ephemeral/home/dataset/test/DCM \
    --save_dir /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/ensemble_preds \
    --resize 1024 \
    --batch_size 1 \
    --use_tta