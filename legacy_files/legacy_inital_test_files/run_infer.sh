#!/bin/bash

MODEL=./checkpoints/Baseline/best_2epoch_0.2388.pt
IMAGE_ROOT=/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/data/test/DCM
OUT=test_output.csv

python inference.py $MODEL \
  --image_root $IMAGE_ROOT \
  --thr 0.5 \
  --resize 1024 \
  --output $OUT


# TODO Task : https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-04-lv3/blob/main/inference.sh