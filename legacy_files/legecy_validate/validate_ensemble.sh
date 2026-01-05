#!/bin/bash

# ============================================================================
# Ensemble Validation 스크립트
# 앙상블 결과 CSV와 GT를 비교하여 Dice score를 계산합니다.
# ============================================================================

# ============================================================================
# 변수 설정 (여기서 수정하세요)
# ============================================================================

# 앙상블 결과 CSV 파일 경로
ENSEMBLE_CSV="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/ensemble/idx7.csv"

# Validation 데이터 경로
IMAGE_ROOT="/data/ephemeral/home/dataset/train/DCM"
LABEL_ROOT="/data/ephemeral/home/dataset/train/outputs_json"

# Python 스크립트 경로
VALIDATE_SCRIPT="/data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/validate_ensemble.py"

# ============================================================================
# Validation 실행
# ============================================================================

python "${VALIDATE_SCRIPT}" \
    "${ENSEMBLE_CSV}" \
    --label_root "${LABEL_ROOT}" \
    --image_root "${IMAGE_ROOT}"

