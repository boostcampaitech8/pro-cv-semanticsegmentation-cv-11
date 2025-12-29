#!/bin/bash

# 여러 config 파일로 학습을 순차적으로 실행하는 스크립트
# 사용법: ./scripts/train_multiple.sh

# 프로젝트 루트 디렉토리
PROJECT_ROOT="/data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11"
cd $PROJECT_ROOT

# 학습할 config 파일 목록 (순서대로 실행)
CONFIGS=(
    "configs/hrnet_w18_config.yaml"
    "configs/hrnet_w48_config.yaml"
    # "configs/unetplusplus_config.yaml"
    # "configs/segformer_config.yaml"
    # 추가 config 파일들...
)

# 로그 디렉토리
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p $LOG_DIR

# 각 config로 학습 실행
for config in "${CONFIGS[@]}"; do
    if [ ! -f "$config" ]; then
        echo "⚠️  Config 파일을 찾을 수 없습니다: $config"
        continue
    fi
    
    # config 파일명에서 확장자 제거하여 로그 파일명 생성
    config_name=$(basename "$config" .yaml)
    log_file="$LOG_DIR/train_${config_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "=========================================="
    echo "🚀 학습 시작: $config"
    echo "📝 로그 파일: $log_file"
    echo "=========================================="
    
    # 학습 실행 (로그 파일에 출력 저장)
    python train.py --config "$config" 2>&1 | tee "$log_file"
    
    # 종료 코드 확인
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ 학습 완료: $config"
    else
        echo "❌ 학습 실패: $config"
        echo "⚠️  다음 config로 계속 진행합니다..."
    fi
    
    echo ""
done

echo "=========================================="
echo "🎉 모든 학습 작업이 완료되었습니다!"
echo "=========================================="

