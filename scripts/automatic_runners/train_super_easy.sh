#!/bin/bash

# 초초간단 버전
python train.py --config configs/hrnet_w18_config.yaml & python train.py --config configs/hrnet_w48_config.yaml &