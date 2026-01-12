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
    "/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/ens_candidates/150_2_team_hrnet_w48-2048-best_88epoch_0.9744.pt"
    "/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/ens_candidates/153_best_76epoch_0.9753.pt"
    "/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/ens_candidates/155_hrnet_w64_1024_crop-best_23epoch_0.9805.pt"
)

# 모델별 가중치 (MODELS 순서와 동일하게, None이면 동일 가중치)
# 기존: (0.27 0.33 0.20 0.20) + 새 모델 0.20 = 합 1.2
# 정규화: 각각을 1.2로 나눔 → (0.225 0.275 0.1667 0.1667 0.1667)
WEIGHTS=(0.45 0.45 0.1)
# WEIGHTS=()  # 빈 배열이면 동일 가중치 사용

# Crop 모델 인덱스 (클래스별 선택적 앙상블용)
# Wrist class 8개는 이 모델만 사용, 나머지 클래스는 다른 모델들만 앙상블
# None이거나 주석 처리하면 기존처럼 모든 클래스에 대해 일반 앙상블
# ROP_MODEL_IDX=0
CROP_MODEL_IDX=""  # 사용하지 않으면 비워두기

# Crop 모드 사용 여부
# true: crop 모델에 crop된 이미지 입력 (full 모델로 wrist ROI 추출 후 crop)
# false: crop 모델에 full 이미지 입력 (기존 방식)
USE_CROP_MODE=false
# USE_CROP_MODE=true

# 이미지 resize 크기 (각 모델마다 다른 크기 지정 가능)
# 방법 1: 배열로 선언 (권장)
#   RESIZE=(1024 1024 1536) - 모델 3개에 각각 1024, 1024, 1536 적용
#   RESIZE=(1024) - 모든 모델에 1024 적용
# 방법 2: 스칼라로 선언 (자동으로 배열로 변환됨)
#   RESIZE=1024 - 모든 모델에 1024 적용
RESIZE=(2048 2048 2048)

# 모델 구조 정보 JSON 파일 (state_dict만 저장된 모델이 있는 경우) 
# 형식: [null, null, {...}, null] - 각 모델에 대한 config (null이면 전체 모델로 간주) 
# 예시: [{"model_name": "UnetPlusPlus", "encoder_name": "se_resnext101_32x4d", 
#         "encoder_weights": null, "in_channels": 3, "classes": 29}] 
# MODEL_CONFIGS="${PROJECT_ROOT}/scripts/ensemble/model_configs.json" 
MODEL_CONFIGS=""  # state_dict 모델이 없으면 비워두기 

# 이미지 루트 경로
# Validation 모드일 때는 train set의 이미지 경로를 사용해야 함 (test set이 아님!)
# IMAGE_ROOT="/data/ephemeral/home/dataset/test/DCM"  # test set (inference용)
IMAGE_ROOT="/data/ephemeral/home/dataset/train/DCM"  # train set (validation용)

# Validation 모드 설정 (CSV 저장 없이 바로 validation 수행)
# USE_VALIDATE=true로 설정하면 앙상블 결과를 바로 GT와 비교하여 Dice score 출력
# USE_VALIDATE=false
USE_VALIDATE=true

# Validation 데이터 경로 (USE_VALIDATE=true일 때 필요)
LABEL_ROOT="/data/ephemeral/home/dataset/train/outputs_json"
VAL_FOLD=0

# TTA 사용 여부
USE_TTA=true
# USE_TTA=false

# 출력 디렉토리 및 파일명
OUTPUT_DIR="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/outputs/ens_candidates"
# OUTPUT_FILE="${OUTPUT_DIR}/ensemble_$(date +%Y%m%d_%H%M%S).csv"

# 직접 지정해서 쓰시는게 편할 거에요.
OUTPUT_FILE="${OUTPUT_DIR}/idx10-3_thr_2.csv"

# Validation 로그 파일 경로 (USE_VALIDATE=true일 때 사용)
LOG_DIR="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/logs/ensemble"
LOG_FILE="${LOG_DIR}/validation_$(date +%Y%m%d_%H%M%S).log"
# LOG_FILE="${LOG_DIR}/validation_idx8.log"  # 직접 지정도 가능  

# 기본 threshold
THR=0.5

# 클래스별 threshold JSON 파일 경로 (선택사항)
# 지정하면 해당 JSON 파일의 threshold를 사용하고, 지정하지 않으면 THR 값을 모든 클래스에 적용
# Validation 결과 기반 최적화: 낮은 Dice score 클래스는 낮은 threshold, 높은 클래스는 높은 threshold
THR_DICT="${PROJECT_ROOT}/configs/class_thresholds/class_thresholds_wrist_04_others_06.json"
# THR_DICT=""  # 사용하지 않으면 비워두기

# 배치 크기
BATCH_SIZE=3

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
# model_configs가 있으면 --model_configs 추가
MODEL_CONFIGS_FLAG=""
if [ -n "${MODEL_CONFIGS}" ] && [ -f "${MODEL_CONFIGS}" ]; then
    MODEL_CONFIGS_FLAG="--model_configs ${MODEL_CONFIGS}"
fi

# Validation 모드 플래그
VALIDATE_FLAG=""
if [ "$USE_VALIDATE" = true ]; then
    # Validation 로그 디렉토리 생성
    mkdir -p "${LOG_DIR}"
    VALIDATE_FLAG="--validate --label_root ${LABEL_ROOT} --val_fold ${VAL_FOLD} --log_file ${LOG_FILE}"
    # Validation 모드에서는 output 파일 불필요 (CSV 저장 안 함)
    OUTPUT_FLAG=""
else
    OUTPUT_FLAG="--output ${OUTPUT_FILE}"
fi

# Python 명령어 구성
PYTHON_ARGS=(
    "${ENSEMBLE_SCRIPT}"
    --models "${MODELS[@]}"
)

if [ ${#WEIGHTS[@]} -gt 0 ]; then
    PYTHON_ARGS+=(--weights "${WEIGHTS[@]}")
fi

if [ -n "${TTA_FLAG}" ]; then
    PYTHON_ARGS+=(${TTA_FLAG})
fi

PYTHON_ARGS+=(--image_root "${IMAGE_ROOT}")

if [ -n "${OUTPUT_FLAG}" ]; then
    PYTHON_ARGS+=(${OUTPUT_FLAG})
fi

PYTHON_ARGS+=(--thr "${THR}")

# threshold JSON 파일이 지정된 경우 추가
if [ -n "${THR_DICT}" ] && [ -f "${THR_DICT}" ]; then
    PYTHON_ARGS+=(--thr_dict "${THR_DICT}")
fi

# Crop 모델 인덱스가 지정된 경우 추가
if [ -n "${CROP_MODEL_IDX}" ] && [ "${CROP_MODEL_IDX}" != "" ]; then
    PYTHON_ARGS+=(--crop_model_idx "${CROP_MODEL_IDX}")
fi

# Crop 모드가 활성화된 경우 추가
if [ "$USE_CROP_MODE" = true ]; then
    PYTHON_ARGS+=(--crop_mode)
fi

PYTHON_ARGS+=(--resize "${RESIZE[@]}")
PYTHON_ARGS+=(--batch_size "${BATCH_SIZE}")

if [ -n "${MODEL_CONFIGS_FLAG}" ]; then
    PYTHON_ARGS+=(${MODEL_CONFIGS_FLAG})
fi

if [ -n "${VALIDATE_FLAG}" ]; then
    PYTHON_ARGS+=(${VALIDATE_FLAG})
fi

# 실행
python "${PYTHON_ARGS[@]}"
