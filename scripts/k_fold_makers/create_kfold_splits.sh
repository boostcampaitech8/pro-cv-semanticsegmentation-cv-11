#!/bin/bash

# K-fold 분할을 파일 시스템에 저장하는 스크립트
# fold0만 생성 (--target_fold 0)

python /data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/scripts/create_kfold_splits.py \
    --image_root /data/ephemeral/home/dataset/train/DCM \
    --label_root /data/ephemeral/home/dataset/train/outputs_json \
    --output_root /data/ephemeral/home/dataset \
    --n_splits 5 \
    --target_fold 0

