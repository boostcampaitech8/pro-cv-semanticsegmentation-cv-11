# Configs 폴더

이 폴더는 모델 학습 및 추론에 사용되는 설정 파일들을 관리합니다.

## 폴더 구조

```
configs/
├── base_config.yaml          # 기본 설정 템플릿
├── hrnet_w18_config.yaml     # HRNet-W18 학습 설정
├── hrnet_w18_unetpp_smp.yaml # HRNet-W18 + UnetPlusPlus (SMP) 설정
├── hrnet_w48_config.yaml     # HRNet-W48 학습 설정
├── mmseg/                    # mmsegmentation 전용 설정 파일들
│   ├── hrnet_w18_fcn.py      # HRNet-W18 + FCNHead 설정
│   ├── hrnet_w18_ocr.py      # HRNet-W18 + OCRHead 설정
│   ├── hrnet_w48_fcn.py      # HRNet-W48 + FCNHead 설정
│   └── README_HRNet.md       # mmseg 라이브러리 및 HRNet 설치&사용 가이드
```

## 주요 파일 설명

### 기본 설정 파일

- **`base_config.yaml`**: 모든 설정 파일의 기본 템플릿. 공통 설정값들을 정의합니다.
  - 데이터 경로, 모델 기본 파라미터, 학습 하이퍼파라미터 등

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

### mmseg 폴더

mmsegmentation 라이브러리 전용 설정 파일들을 포함합니다.

- **`hrnet_w18_fcn.py`**: HRNet-W18 + FCNHead 설정 (Python 형식)
  - mmsegmentation의 표준 config 형식
  - `EncoderDecoder` 구조 사용

- **`hrnet_w18_ocr.py`**: HRNet-W18 + OCRHead 설정
  - `CascadeEncoderDecoder` 구조 사용
  - FCNHead (auxiliary) + OCRHead (main) 조합

- **`hrnet_w48_fcn.py`**: HRNet-W48 + FCNHead 설정

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
  config_path: configs/mmseg/hrnet_w18_fcn.py
  num_classes: 29

# 배치 사이즈
train_batch_size: 2
val_batch_size: 2

# 학습 하이퍼파라미터
lr: 0.001 또는 0.003
weight_decay: 1e-6
# ... 기타 설정
```

## 주의사항

- mmseg 폴더의 Python 설정 파일은 mmsegmentation 라이브러리 전용입니다.

