#!/bin/bash

# ============================================================================
# 통합 Ensemble 스크립트
# 사용 방법: 위쪽 변수들을 수정하고, 원하는 케이스의 주석을 해제하여 사용
# ============================================================================

# ============================================================================
# 변수 설정 (여기서 수정하세요)
# ============================================================================

# 프로젝트 루트 디렉토리
PROJECT_ROOT="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11"

# 앙상블할 모델 경로들 (여러 개 가능)
MODELS=(
    "/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W48/hrnet_w48-best_30epoch_0.9727.pt"
    "/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W18/251230/hrnet_w18_aug_elastic_3_stage-best_28epoch_0.9720.pt"
    "/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W18/251230/hrnet_w18_ocr-best_30epoch_0.9714.pt"
    "/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W18/251228_unetpp_simple/hrnet_w18_unetpp_batch_size_2-30epoch_0.9713.pt"
)

# 모델별 가중치 (MODELS 순서와 동일하게, None이면 동일 가중치)
WEIGHTS=(0.27 0.27 0.23 0.23)
# WEIGHTS=()  # 빈 배열이면 동일 가중치 사용

# TTA 사용 여부
USE_TTA=true
# USE_TTA=false

# 이미지 루트 경로
IMAGE_ROOT="/data/ephemeral/home/dataset/test/DCM"

# 출력 디렉토리 및 파일명
OUTPUT_DIR="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/outputs/ens_candidates"
# OUTPUT_FILE="${OUTPUT_DIR}/ensemble_$(date +%Y%m%d_%H%M%S).csv"

# 직접 지정해서 쓰시는게 편할 거에요.
OUTPUT_FILE="${OUTPUT_DIR}/hrneHRNet_W48-augelastic-ocr-unetpp.csv"  

# 기본 threshold
THR=0.5

# 이미지 resize 크기 (각 모델마다 다른 크기 지정 가능)
# 방법 1: 배열로 선언 (권장)
#   RESIZE=(1024 1024 1536) - 모델 3개에 각각 1024, 1024, 1536 적용
#   RESIZE=(1024) - 모든 모델에 1024 적용
# 방법 2: 스칼라로 선언 (자동으로 배열로 변환됨)
#   RESIZE=1024 - 모든 모델에 1024 적용
RESIZE=(1024)

# 배치 크기
BATCH_SIZE=1

# Python 스크립트 경로
ENSEMBLE_SCRIPT="${PROJECT_ROOT}/scripts/ensemble/ensemble.py"

# ============================================================================
# 출력 디렉토리 생성
# ============================================================================
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# 앙상블 실행
# ============================================================================

# TTA 플래그
TTA_FLAG=""
if [ "$USE_TTA" = true ]; then
    TTA_FLAG="--use_tta"
fi

# Python 스크립트 실행
# 가중치가 있으면 --weights 추가, 없으면 생략
if [ ${#WEIGHTS[@]} -gt 0 ]; then
    python "${ENSEMBLE_SCRIPT}" \
        --models "${MODELS[@]}" \
        --weights "${WEIGHTS[@]}" \
        ${TTA_FLAG} \
        --image_root "${IMAGE_ROOT}" \
        --output "${OUTPUT_FILE}" \
        --thr "${THR}" \
        --resize "${RESIZE[@]}" \
        --batch_size "${BATCH_SIZE}"
else
    python "${ENSEMBLE_SCRIPT}" \
        --models "${MODELS[@]}" \
        ${TTA_FLAG} \
        --image_root "${IMAGE_ROOT}" \
        --output "${OUTPUT_FILE}" \
        --thr "${THR}" \
        --resize "${RESIZE[@]}" \
        --batch_size "${BATCH_SIZE}"
fi
