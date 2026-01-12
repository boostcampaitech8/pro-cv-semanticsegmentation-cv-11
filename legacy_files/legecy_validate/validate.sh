#!/bin/bash

# ============================================================================
# Validation 스크립트
# 모델의 validation set에 대한 Dice score를 계산합니다.
# ============================================================================

# ============================================================================
# 변수 설정 (여기서 수정하세요)
# ============================================================================

# 모델 경로
MODEL_PATH="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W18/251231/hrnet_w18_ocr_multiplier_1.0_cosinewarmup-best_30epoch_0.9720.pt"

# Validation 데이터 경로
IMAGE_ROOT="/data/ephemeral/home/dataset/fold0/val/DCM"
LABEL_ROOT="/data/ephemeral/home/dataset/fold0/val/outputs_json"

# Threshold 설정
THR=0.5
THR_DICT=""  # 클래스별 threshold 사용 시 경로 지정
# THR_DICT="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/configs/class_thresholds/class_thresholds_hard.json"

# 기타 설정
RESIZE=1024
BATCH_SIZE=1
VAL_FOLD=0

# TTA 사용 여부
USE_TTA=false
# USE_TTA=true

# Python 스크립트 경로
VALIDATE_SCRIPT="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/validate.py"

# ============================================================================
# Validation 실행
# ============================================================================

TTA_FLAG=""
if [ "$USE_TTA" = true ]; then
    TTA_FLAG="--use_tta"
fi

THR_DICT_FLAG=""
if [ -n "$THR_DICT" ]; then
    THR_DICT_FLAG="--thr_dict ${THR_DICT}"
fi

python "${VALIDATE_SCRIPT}" \
    "${MODEL_PATH}" \
    --image_root "${IMAGE_ROOT}" \
    --label_root "${LABEL_ROOT}" \
    --thr "${THR}" \
    ${THR_DICT_FLAG} \
    --resize "${RESIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --val_fold "${VAL_FOLD}" \
    ${TTA_FLAG}

