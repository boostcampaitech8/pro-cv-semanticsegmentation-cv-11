#!/bin/bash

# 초간단 버전
for config in configs/hrnet_w18_config.yaml configs/hrnet_w48_config.yaml; do python train.py --config $config; done