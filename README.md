# 의료 영상 세그멘테이션 프로젝트
### README.md last modified: 251231

손목 X-ray 영상에서 29개의 뼈 구조를 자동으로 분할(세그멘테이션)하는 딥러닝 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 의료 영상 분석을 위한 세그멘테이션 모델을 학습하고 추론하는 시스템입니다. 손목 X-ray 이미지를 입력받아 29개의 뼈 구조(손가락 뼈, 손목 뼈 등)를 픽셀 단위로 정확하게 분할합니다.

### 주요 기능

- **모델 학습**: 다양한 아키텍처(HRNet, UNet++, SegFormer 등) 지원
- **추론**: 학습된 모델로 테스트 이미지에 대한 예측 수행
- **앙상블**: 여러 모델의 예측 결과를 결합하여 성능 향상
- **시각화**: 예측 결과를 이미지로 시각화하여 분석
- **K-Fold 교차 검증**: 데이터를 분할하여 모델 성능 평가

## 빠른 시작

### 1. 학습 (Training)

설정 파일을 작성한 후 학습을 시작합니다:

```bash
# 1. 모델 config 파일 수정
vi configs/hrnet_w18_config.yaml

# 2. 학습 스크립트 수정
vi train.sh

# 3. 학습 실행
./train.sh
```

또는 직접 Python 스크립트 실행:

```bash
python train.py --config configs/hrnet_w18_config.yaml
```

### 2. 추론 (Inference)

학습된 모델로 테스트 이미지에 대한 예측을 수행합니다:

```bash
# 1. 추론 스크립트 수정
vi inference.sh

# 2. 추론 실행
./inference.sh
```

### 3. 앙상블

여러 모델의 예측 결과를 결합합니다:

```bash
# 1. 앙상블 스크립트 수정
vi scripts/ensemble/ensemble.sh

# 2. 앙상블 실행
./scripts/ensemble/ensemble.sh
```

## 프로젝트 구조

```
pro-cv-semanticsegmentation-cv-11/
│
├── train.py                  # 학습 메인 스크립트
├── train.sh                  # 학습 실행 스크립트
├── inference.py              # 추론 메인 스크립트
├── inference.sh              # 추론 실행 스크립트
├── trainer.py                # 학습 로직 구현
├── dataset.py                # 데이터셋 정의
│
├── configs/                  # 설정 파일들
│   ├── *.yaml                # 모델별 학습 설정
│   ├── mmseg_config_py_files/  # mmsegmentation 설정
│   ├── class_thresholds/      # 클래스별 threshold 설정
│   ├── class_weights/         # 클래스별 가중치 설정
│   └── personal_configs/      # 개인별 설정 파일들
│
├── models/                   # 모델 아키텍처 정의
│   ├── hrnet.py              # HRNet 모델
│   ├── unetplusplus.py       # UNet++ 모델
│   └── model_picker.py       # 모델 선택 관리
│
├── loss/                     # Loss 함수들
│   ├── bce.py                # Binary Cross Entropy
│   ├── dice.py               # Dice Loss
│   └── loss_mixer.py         # Loss 조합 관리
│
├── scripts/                  # 보조 스크립트들
│   ├── ensemble/             # 앙상블 관련
│   ├── k_fold_makers/        # K-Fold 데이터 분할
│   ├── visualizer/           # 결과 시각화
│   └── custom_runners/        # 커스텀 실행 스크립트들
│
├── optimizers/               # Optimizer 정의
├── scheduler/                # Learning Rate Scheduler 정의
├── utils/                    # 유틸리티 함수들
├── docs/                     # 문서 파일들
└── legacy_files/             # 레거시 파일들 (참고용)
```

## 주요 폴더 설명

### configs/
모델 학습 및 추론에 필요한 설정 파일들을 관리합니다. YAML 형식으로 모델, 데이터, 학습 하이퍼파라미터 등을 정의합니다.

자세한 내용: [configs/README-configs.md](configs/README-configs.md)

### models/
세그멘테이션 모델 아키텍처를 정의합니다. HRNet, UNet++, SegFormer 등 다양한 모델을 지원합니다.

자세한 내용: [models/README-models.md](models/README-models.md)

### loss/
학습에 사용되는 Loss 함수들을 정의합니다. BCE, Dice, Focal Loss 등 다양한 Loss를 지원하며, 여러 Loss를 조합하여 사용할 수 있습니다.

자세한 내용: [loss/README-loss.md](loss/README-loss.md)

### scripts/
앙상블, 데이터 분할, 시각화 등의 보조 작업을 수행하는 스크립트들을 관리합니다.

자세한 내용: [scripts/README-scripts.md](scripts/README-scripts.md)

## 주요 기능 상세

### 3-Stage Loss Switching

학습 과정에서 모델 성능에 따라 Loss 함수의 비율을 동적으로 변경하는 기법입니다. 초기에는 안정적인 학습을 위해 BCE Loss 중심으로 시작하고, 모델이 성장함에 따라 Dice Loss 비중을 점진적으로 높여 최종 성능을 최적화합니다.

자세한 내용: [configs/What_is_3_stage_loss_switching.md](configs/What_is_3_stage_loss_switching.md)

### Test Time Augmentation (TTA)

추론 시 데이터 증강을 적용하여 성능을 향상시키는 기법입니다. HorizontalFlip 등의 변환을 적용하고 결과를 평균하여 최종 예측을 생성합니다.

### 앙상블

여러 모델의 예측 결과를 가중 평균하여 최종 결과를 생성합니다. 각 모델의 강점을 결합하여 단일 모델보다 향상된 성능을 달성할 수 있습니다.

## 요구사항

- Python 3.8+
- PyTorch
- mmsegmentation (mmseg 기반 HRNet 사용 시)
- Segmentation Models PyTorch (UNet++ 사용 시)
- 기타 필수 라이브러리 (requirements.txt 참고)

## 참고 문서

각 폴더의 README 파일에서 더 자세한 정보를 확인할 수 있습니다:

- [configs/README-configs.md](configs/README-configs.md) - 설정 파일 가이드
- [models/README-models.md](models/README-models.md) - 모델 아키텍처 설명
- [loss/README-loss.md](loss/README-loss.md) - Loss 함수 설명
- [scripts/README-scripts.md](scripts/README-scripts.md) - 스크립트 사용법

## 라이선스

이 프로젝트는 대회 참가를 위한 프로젝트입니다.

