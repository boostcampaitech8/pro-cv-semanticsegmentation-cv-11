#!/bin/bash

# 출력 디렉토리 생성
mkdir -p /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/outputs/HRNet_W18

python /data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/inference.py \
    /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W18/251228_unetpp_simple/hrnet_w18_unetpp_with_smp_elastic_aug-29epoch_0.9710.pt \
    --image_root /data/ephemeral/home/dataset/test/DCM \
    --thr 0.5 \
    --output /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/outputs/HRNet_W18/hrnet_w18-unetpp_with_smp_elastic_aug-29epoch_0.9710.csv \
    --resize 1024