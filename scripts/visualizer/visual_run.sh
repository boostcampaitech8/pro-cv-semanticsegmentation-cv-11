#!/bin/bash

# ============================================================================
# 통합 Visualization 스크립트
# 사용 방법: 위쪽 변수들을 수정하고, 원하는 케이스의 주석을 해제하여 사용
# ============================================================================

# ============================================================================
# 변수 설정 (여기서 수정하세요)
# ============================================================================

# 프로젝트 루트 디렉토리
PROJECT_ROOT="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11"

# 시각화할 CSV 파일 경로
CSV_FILE="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/outputs/HRNet_W18/hrnet_w18-bce03_dice07.csv"

### 주의!!! ###
### k-fold maker로 이미 데이터 셋을 분할해서 파일 형태로 저장했어야 아래 스크립트들이 실행 가능합니다. ###
### /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/scripts/k_fold_makers/create_kfold_splits.sh를 먼저 실행해 주세요. ###

# 이미지 루트 경로
IMAGE_ROOT="/data/ephemeral/home/dataset/fold0/val/DCM/"

# 라벨 루트 경로 (GT 포함 시각화용)
LABEL_ROOT="/data/ephemeral/home/dataset/fold0/val/outputs_json/"

# 시각화할 ID 목록 (공백으로 구분)
IDS="ID004 ID009 ID014 ID313 ID338 ID543"

# 시각화할 클래스 목록 (공백으로 구분, 빈 값이면 모든 클래스)
ONLY_CLASSES="Pisiform Triquetrum"
# ONLY_CLASSES=""  # 모든 클래스 시각화

# 저장 디렉토리
SAVE_ROOT="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/visual/vis_results"

# Python 스크립트 경로
VISUAL_SCRIPT="${PROJECT_ROOT}/visual/visual.py"
VISUAL_WITH_GT_SCRIPT="${PROJECT_ROOT}/visual/visual_with_GT.py"

# ============================================================================
# 저장 디렉토리 생성
# ============================================================================
mkdir -p "${SAVE_ROOT}"

# ============================================================================
# 1. 기본 시각화 (GT 없음)
# ============================================================================
python "${VISUAL_SCRIPT}" \
  --csv "${CSV_FILE}" \
  --image_root "${IMAGE_ROOT}" \
  --ids ${IDS} \
  --save_root "${SAVE_ROOT}" \
  ${ONLY_CLASSES:+--only-class ${ONLY_CLASSES}}

# ============================================================================
# 2. GT 포함 시각화 (validation set 사용)
# ============================================================================
# python "${VISUAL_WITH_GT_SCRIPT}" \
#   --csv "${CSV_FILE}" \
#   --image_root "${IMAGE_ROOT}" \
#   --label_root "${LABEL_ROOT}" \
#   --ids ${IDS} \
#   --save_root "${SAVE_ROOT}" \
#   ${ONLY_CLASSES:+--only-class ${ONLY_CLASSES}}
