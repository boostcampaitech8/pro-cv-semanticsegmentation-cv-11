#!/bin/bash

# ============================================================================
# 통합 Train 스크립트
# 사용 방법: 위쪽 변수들을 수정하고, 원하는 케이스의 주석을 해제하여 사용
# ============================================================================

# ============================================================================
# 변수 설정 (여기서 수정하세요)
# ============================================================================

# 프로젝트 루트 디렉토리
PROJECT_ROOT="/data/ephemeral/home/pro-cv-semanticsegmentation-cv-11"

# 학습할 config 파일 목록 (순서대로 실행)
# 단일 config만 사용하려면 하나만 남기고 나머지는 주석 처리
CONFIGS=(
    "/data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/configs/configs_example/260106/hrnet_w64_1024_crop.yaml"
    # "/data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/configs/configs_example/260103/hrnet_w48_crop_post_training.yaml"
    # "/data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/configs/configs_example/260101/hrnet_w48_ocr_cosinewarmup_50e.yaml"
    # "/data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/configs/configs_example/251231/hrnet_w18_ocr_multiplier_0.5_thr_0.8.yaml"
    # "/data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/configs/251230/hrnet_w18_stage_threshold_high_5050.yaml"
)

# 로그 디렉토리 (절대경로 또는 상대경로)
LOG_DIR="${PROJECT_ROOT}/logs/260106"

# Python 스크립트 경로
TRAIN_SCRIPT="${PROJECT_ROOT}/train.py"

# ============================================================================
# 프로젝트 루트로 이동
# ============================================================================
cd "${PROJECT_ROOT}"

# ============================================================================
# 로그 디렉토리 생성
# ============================================================================
mkdir -p "${LOG_DIR}"

# ============================================================================
# 각 config로 학습 실행
# ============================================================================
for config in "${CONFIGS[@]}"; do
    # 빈 요소나 주석 처리된 항목 건너뛰기
    if [ -z "$config" ] || [[ "$config" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # config 파일 존재 확인
    if [ ! -f "$config" ]; then
        echo "⚠️  Config 파일을 찾을 수 없습니다: $config"
        continue
    fi
    
    # config 파일명에서 확장자 제거하여 로그 파일명 생성
    config_name=$(basename "$config" .yaml)
    log_file="${LOG_DIR}/train_${config_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "=========================================="
    echo "🚀 학습 시작: $config"
    echo "📝 로그 파일: $log_file"
    echo "=========================================="
    
    # 학습 실행 (로그 파일에 출력 저장)
    python "${TRAIN_SCRIPT}" --config "$config" 2>&1 | tee "$log_file"
    
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
