#!/bin/bash

# 저장된 확률값들을 앙상블하여 CSV 생성하는 스크립트
# 사용 시 아래 인자들을 수정하여 사용하세요

# 출력 디렉토리 생성
mkdir -p /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/ensemble

python /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/scripts/ensemble/ensemble_predictions.py \
    --pred_dir /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/ensemble_preds \
    --model_names hrnet_w18_aug_elastic_3_stage-best_28epoch_0.9720 hrnet_w18_unetpp_with_smp_elastic_aug-29epoch_0.9710 \
    --output_dir /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/ensemble \
    --thr 0.5 \
    --weights 0.5 0.5

