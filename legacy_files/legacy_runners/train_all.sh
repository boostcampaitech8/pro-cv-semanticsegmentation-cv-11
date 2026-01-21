#!/bin/bash

# 간단한 버전: configs 디렉토리의 모든 yaml 파일로 학습
# 사용법: ./scripts/train_all.sh

PROJECT_ROOT="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11"
cd $PROJECT_ROOT

# configs 디렉토리의 모든 yaml 파일 찾기 (base_config.yaml 제외)
CONFIGS=$(find configs -name "*.yaml" -type f ! -name "base_config.yaml" | sort)

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p $LOG_DIR

for config in $CONFIGS; do
    config_name=$(basename "$config" .yaml)
    log_file="$LOG_DIR/train_${config_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "🚀 학습 시작: $config"
    python train.py --config "$config" 2>&1 | tee "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ 완료: $config"
    else
        echo "❌ 실패: $config"
    fi
    echo ""
done

echo "🎉 모든 학습 완료!"

