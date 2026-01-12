#!/bin/bash

# ============================================================================
# K-fold 분할 스크립트
# 사용 방법: 위쪽 변수들을 수정하여 사용
# ============================================================================

# ============================================================================
# 변수 설정 (여기서 수정하세요)
# ============================================================================

# 프로젝트 루트 디렉토리
PROJECT_ROOT="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11"

# 이미지 루트 경로
IMAGE_ROOT="/data/ephemeral/home/dataset/train/DCM"

# 라벨 루트 경로
LABEL_ROOT="/data/ephemeral/home/dataset/train/outputs_json"

# 출력 루트 경로 (fold0, fold1 등의 폴더가 생성될 위치)
OUTPUT_ROOT="/data/ephemeral/home/dataset"

# K-fold 분할 개수
N_SPLITS=5

# 생성할 fold 번호 (0부터 시작, None이면 모든 fold 생성)
TARGET_FOLD=0
# TARGET_FOLD=""  # 빈 값이면 모든 fold 생성

# Python 스크립트 경로
K_FOLD_SCRIPT="${PROJECT_ROOT}/scripts/k_fold_makers/create_kfold_splits.py"

# ============================================================================
# K-fold 분할 실행
# ============================================================================

# TARGET_FOLD 인자 생성
TARGET_FOLD_ARG=""
if [ -n "$TARGET_FOLD" ]; then
    TARGET_FOLD_ARG="--target_fold ${TARGET_FOLD}"
fi

# Python 스크립트 실행
python "${K_FOLD_SCRIPT}" \
    --image_root "${IMAGE_ROOT}" \
    --label_root "${LABEL_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --n_splits "${N_SPLITS}" \
    ${TARGET_FOLD_ARG}
