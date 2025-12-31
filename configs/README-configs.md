# Configs 폴더

모델 학습 및 추론에 사용되는 설정 파일들을 관리하는 폴더입니다.

## 폴더 구조

```
configs/
├── base_config.yaml                    # 기본 설정 템플릿
├── hrnet_w18_config.yaml               # HRNet-W18 학습 설정
├── hrnet_w18_unetpp_smp.yaml           # HRNet-W18 + UnetPlusPlus (SMP) 설정
├── hrnet_w48_config.yaml               # HRNet-W48 학습 설정
├── class_thresholds/                   # 클래스별 threshold 설정 파일들
│   ├── class_thresholds.json
│   └── class_thresholds_hard.json
├── mmseg_config_py_files/              # mmsegmentation 전용 설정 파일들
│   ├── hrnet_w18_fcn.py                # HRNet-W18 + FCNHead 설정
│   ├── hrnet_w18_ocr.py                # HRNet-W18 + OCRHead 설정
│   ├── hrnet_w48_fcn.py                # HRNet-W48 + FCNHead 설정
│   └── README_HRNet.md                 # mmseg 라이브러리 및 HRNet 설치&사용 가이드
├── configs_example/                     # 예시 설정 파일들
│   └── ...
├── What_is_3_stage_loss_switching.md   # 3-Stage Loss Switching 설명 문서
└── README-configs.md                   # 이 파일
```

## 주요 파일 설명

### 기본 설정 파일

- **`base_config.yaml`**: 모든 설정 파일의 기본 템플릿
  - 데이터 경로, 모델 기본 파라미터, 학습 하이퍼파라미터 등 공통 설정값 정의

### 모델별 설정 파일

- **`hrnet_w18_config.yaml`**: HRNet-W18 모델 학습 설정
  - mmsegmentation의 HRNet-W18 + FCNHead 조합
  - 경량 모델로 빠른 학습과 적은 메모리 사용

- **`hrnet_w18_unetpp_smp.yaml`**: HRNet-W18 + UnetPlusPlus (SMP) 조합
  - SMP 라이브러리의 UnetPlusPlus 디코더 사용
  - `tu-hrnet_w18` 인코더 사용

- **`hrnet_w48_config.yaml`**: HRNet-W48 모델 학습 설정
  - 더 큰 모델로 높은 성능 기대
  - 더 많은 메모리 필요

### class_thresholds/

클래스별 threshold 설정 파일들을 포함합니다.

- **`class_thresholds.json`**: 기본 클래스별 threshold 값
- **`class_thresholds_hard.json`**: 더 엄격한 threshold 값

추론 시 각 클래스에 대한 최적 threshold 값을 사용하여 이진화합니다.

### mmseg_config_py_files/

mmsegmentation 라이브러리 전용 설정 파일들을 포함합니다.

- **`hrnet_w18_fcn.py`**: HRNet-W18 + FCNHead 설정 (Python 형식)
  - mmsegmentation의 표준 config 형식
  - `EncoderDecoder` 구조 사용

- **`hrnet_w18_ocr.py`**: HRNet-W18 + OCRHead 설정
  - `CascadeEncoderDecoder` 구조 사용
  - FCNHead (auxiliary) + OCRHead (main) 조합

- **`hrnet_w48_fcn.py`**: HRNet-W48 + FCNHead 설정

### configs_example/

다양한 실험 설정 예시 파일들을 포함합니다.

### What_is_3_stage_loss_switching.md

3-Stage Loss Switching 기법에 대한 상세 설명 문서입니다. 학습 과정에서 loss 함수의 비율을 동적으로 변경하는 방법을 설명합니다.

## 설정 파일 사용법

### 학습 실행

```bash
# HRNet-W18 학습
python train.py --config configs/hrnet_w18_config.yaml

# HRNet-W18 + UnetPlusPlus 학습
python train.py --config configs/hrnet_w18_unetpp_smp.yaml
```

### 설정 파일 구조

모든 YAML 설정 파일은 다음 구조를 따릅니다:

```yaml
# 데이터 경로
image_root: /path/to/images
label_root: /path/to/labels

# 모델 설정
model_name: HRNet  # 또는 UnetPlusPlus 등
model_parameter:
  config_path: configs/mmseg_config_py_files/hrnet_w18_fcn.py
  num_classes: 29

# 배치 사이즈
train_batch_size: 2
val_batch_size: 2

# 학습 하이퍼파라미터
lr: 0.001
weight_decay: 1e-6
max_epoch: 30

# Loss 설정
loss_name: Mixed
loss_parameter:
  losses:
    - name: BCEWithLogitsLoss
      weight: 0.3
    - name: DiceLoss
      weight: 0.7

# 3-Stage Loss Switching (선택사항)
loss_switch:
  loss_stages:
    stage2:
      enabled: true
      # ...

# Scheduler, Optimizer 등 기타 설정
# ...
```

## 주의사항

- mmseg_config_py_files/의 Python 설정 파일은 mmsegmentation 라이브러리 전용입니다.
- configs_example/ 폴더의 파일들은 참고용 예시입니다.
- class_thresholds/의 JSON 파일은 추론 시 클래스별 threshold를 적용할 때 사용됩니다.
