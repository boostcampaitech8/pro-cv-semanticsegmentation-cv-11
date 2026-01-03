#!/bin/bash

# ============================================================================
# 통합 Inference 스크립트
# 사용 방법: 위쪽 변수들을 수정하고, 원하는 케이스의 주석을 해제하여 사용
# ============================================================================

# ============================================================================
# 변수 설정 (여기서 수정하세요)
# ============================================================================
MODEL_PATH="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W48/hrnet_w48_cosinewarmup-best_29epoch_0.9731.pt"
IMAGE_ROOT="/data/ephemeral/home/dataset/test/DCM"
OUTPUT_DIR="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/outputs/HRNet_W48"
THR=0.5
THR_DICT="/data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/scripts/basic_runners/class_thresholds/class_thresholds_hard.json"
RESIZE=1024
BATCH_SIZE=1

# 출력 파일명 (모델명에서 자동 추출하거나 직접 지정)
# basename: 경로에서 파일명만 추출
# 첫 번째 인자: 전체 경로
# 두 번째 인자 (.pt): 제거할 접미사

# 아래 예시의 경우, hrnet_w18_aug_elastic_3_stage-best_28epoch_0.9720 만 추출된 후,
# /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/outputs/HRNet_W18/hrnet_w18_aug_elastic_3_stage-best_28epoch_0.9720.csv 파일이 생성됨.
MODEL_NAME=$(basename "$MODEL_PATH" .pt)
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}.csv"

# Python 스크립트 경로
INFERENCE_SCRIPT="/data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/inference.py"

# ============================================================================
# 출력 디렉토리 생성
# ============================================================================
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# 1. 기본 Inference (TTA 없음, 기본 threshold 0.5)
# ============================================================================
# python "${INFERENCE_SCRIPT}" \
#     "${MODEL_PATH}" \
#     --image_root "${IMAGE_ROOT}" \
#     --thr "${THR}" \
#     --output "${OUTPUT_FILE}" \
#     --resize "${RESIZE}" \
#     --batch_size "${BATCH_SIZE}"

# ============================================================================
# 2. TTA 사용 Inference (--use_tta 플래그 사용)
# ============================================================================
python "${INFERENCE_SCRIPT}" \
    "${MODEL_PATH}" \
    --image_root "${IMAGE_ROOT}" \
    --thr "${THR}" \
    --use_tta \
    --output "${OUTPUT_DIR}/${MODEL_NAME}_tta.csv" \
    --resize "${RESIZE}" \
    --batch_size "${BATCH_SIZE}"

# ============================================================================
# 3. Class-wise Threshold 사용 Inference
# ============================================================================
# python "${INFERENCE_SCRIPT}" \
#     "${MODEL_PATH}" \
#     --image_root "${IMAGE_ROOT}" \
#     --thr "${THR}" \
#     --thr_dict "${THR_DICT}" \
#     --output "${OUTPUT_DIR}/${MODEL_NAME}_classwise_thr.csv" \
#     --resize "${RESIZE}" \
#     --batch_size "${BATCH_SIZE}"

# ============================================================================
# 4. TTA + Class-wise Threshold 사용 Inference
# ============================================================================
# python "${INFERENCE_SCRIPT}" \
#     "${MODEL_PATH}" \
#     --image_root "${IMAGE_ROOT}" \
#     --thr "${THR}" \
#     --thr_dict "${THR_DICT}" \
#     --use_tta \
#     --output "${OUTPUT_DIR}/${MODEL_NAME}_tta_classwise_thr.csv" \
#     --resize "${RESIZE}" \
#     --batch_size "${BATCH_SIZE}"

# ============================================================================
# 5. Visualization용 Inference (validation set 사용)
# ============================================================================
# VIS_OUTPUT_DIR="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/for_vis/HRNet_W18"
# VIS_IMAGE_ROOT="/data/ephemeral/home/dataset/fold0/val/DCM"
# mkdir -p "${VIS_OUTPUT_DIR}"
# python "${INFERENCE_SCRIPT}" \
#     "${MODEL_PATH}" \
#     --image_root "${VIS_IMAGE_ROOT}" \
#     --thr "${THR}" \
#     --output "${VIS_OUTPUT_DIR}/${MODEL_NAME}_for_vis.csv" \
#     --resize "${RESIZE}" \
#     --batch_size "${BATCH_SIZE}"
