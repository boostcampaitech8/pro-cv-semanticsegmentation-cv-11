#!/bin/bash

set -e   # 중간에 에러 나면 즉시 종료

EXP_DATE=$(date +"%Y%m%d_%H%M")
LOG_DIR=logs/${EXP_DATE}
mkdir -p ${LOG_DIR}

echo "========================================"
echo " SegFormer Scheduler Experiments Start "
echo " Start Time: $(date)"
echo " Logs: ${LOG_DIR}"
echo "========================================"

# -------------------------------
# 1. Single-cycle cosine + warmup
# -------------------------------
echo "[1/2] Single-cycle cosine warmup training"
python train.py \
  --config configs/custom.yaml \
  | tee ${LOG_DIR}/single_cycle.log

echo "[1/2] Done"
echo "----------------------------------------"

# -------------------------------
# 2. Cosine warmup + restart (10epoch)
# -------------------------------
echo "[2/2] Cosine warmup + restart (10 epoch)"
python train.py \
  --config configs/custom2.yaml \
  | tee ${LOG_DIR}/restart_10epoch.log

echo "[2/2] Done"
echo "----------------------------------------"

echo "========================================"
echo " All Experiments Finished "
echo " End Time: $(date)"
echo "========================================"
