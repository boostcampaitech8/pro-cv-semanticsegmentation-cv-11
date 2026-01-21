# Ensemble 폴더

여러 모델의 예측 결과를 앙상블하여 최종 결과를 생성하는 스크립트들입니다.

## 폴더 구조

```
scripts/ensemble/
├── ensemble.py              # 앙상블 Python 스크립트
├── ensemble.sh              # 앙상블 실행 스크립트
├── ensemble_2step_failed/   # 실패한 2-step 앙상블 방법 (참고용)
└── README-ensemble.md        # 이 파일
```

## 주요 파일 설명

### ensemble.py

1-step 앙상블을 수행하는 Python 스크립트입니다.

**주요 기능:**
- 여러 모델을 동시에 로드하여 배치 단위로 앙상블
- 메모리 효율적인 배치 처리 (OOM 방지)
- 가중 평균 지원
- TTA (Test Time Augmentation) 지원
- 클래스별 threshold 적용 가능

**사용법:**
```bash
python scripts/ensemble/ensemble.py \
    --models model1.pt model2.pt model3.pt \
    --weights 0.4 0.3 0.3 \
    --use_tta \
    --image_root /path/to/images \
    --output /path/to/output.csv \
    --thr 0.5 \
    --resize 1024 \
    --batch_size 1
```

**주요 인자:**
- `--models`: 앙상블할 모델 파일 경로들 (여러 개)
- `--weights`: 각 모델의 가중치 (선택사항, 없으면 평균)
- `--use_tta`: TTA 사용 여부
- `--image_root`: 추론할 이미지 루트 경로
- `--output`: 출력 CSV 파일 경로
- `--thr`: 기본 threshold 값
- `--thr_dict`: 클래스별 threshold JSON 파일 경로 (선택사항)
- `--resize`: 이미지 리사이즈 크기
- `--batch_size`: 배치 크기

### ensemble.sh

앙상블을 실행하는 Shell 스크립트입니다.

**주요 특징:**
- 상단에 변수 선언으로 설정 관리
- 여러 모델과 가중치를 배열로 관리
- TTA 사용 여부 설정 가능

**사용법:**
```bash
# 스크립트 상단의 변수들을 수정
MODELS=(
    "/path/to/model1.pt"
    "/path/to/model2.pt"
)

WEIGHTS=(0.6 0.4)
USE_TTA=true

# 실행
./scripts/ensemble/ensemble.sh
```

## 앙상블 방법

### 1-Step 앙상블 (현재 사용 중)

1. 모든 모델을 한 번에 로드
2. 각 배치에 대해:
   - 각 모델로 추론 수행
   - 확률값을 가중 평균으로 앙상블
   - Threshold 적용
   - RLE 인코딩하여 CSV 저장

**장점:**
- 메모리 효율적 (배치 단위 처리)
- 빠른 처리 속도
- 중간 파일 저장 불필요

### 2-Step 앙상블 (실패, 참고용)

`ensemble_2step_failed/` 폴더에 보관되어 있습니다.

1. Step 1: 각 모델의 확률값을 파일로 저장
2. Step 2: 저장된 확률값을 불러와 앙상블

**문제점:**
- 중간 파일이 매우 큼 (130GB+)
- 디스크 공간 부족
- 메모리 부족 (OOM)

## TTA (Test Time Augmentation)

`--use_tta` 플래그를 사용하면 HorizontalFlip을 적용한 TTA를 수행합니다.

- 각 모델에 TTA 적용
- TTA 결과를 평균하여 앙상블
- 최종 앙상블 결과 생성

## 가중 평균

`--weights` 인자로 각 모델의 가중치를 지정할 수 있습니다.

```bash
--models model1.pt model2.pt model3.pt \
--weights 0.5 0.3 0.2
```

가중치를 지정하지 않으면 단순 평균을 사용합니다.

## 클래스별 Threshold

`--thr_dict` 인자로 클래스별 threshold를 적용할 수 있습니다.

```json
{
  "Pisiform": 0.6,
  "Trapezoid": 0.7,
  ...
}
```

## 주의사항

- 여러 모델을 동시에 로드하므로 GPU 메모리를 충분히 확보해야 합니다.
- 배치 크기는 사용 가능한 메모리에 맞게 조정하세요.
- TTA 사용 시 처리 시간이 증가합니다.

