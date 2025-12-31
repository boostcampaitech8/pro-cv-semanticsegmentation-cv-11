# Models 폴더

세그멘테이션 모델 아키텍처를 정의하는 폴더입니다.

## 폴더 구조

```
models/
├── hrnet.py                 # HRNet 모델 (mmsegmentation 기반)
├── unetplusplus.py          # UnetPlusPlus 모델 (SMP 기반)
├── unet3plus.py              # UNet 3+ 모델
├── segformer.py              # SegFormer 모델
├── model_picker.py           # 모델 선택 및 생성 관리
└── README-models.md          # 이 파일
```

## 주요 파일 설명

### 모델 구현

- **`hrnet.py`**: `HRNet` - High-Resolution Network
  - mmsegmentation 라이브러리 기반
  - FCNHead 또는 OCRHead 디코더 지원
  - HRNet-W18, HRNet-W48 등 다양한 버전 지원

- **`unetplusplus.py`**: `UnetPlusPlus` - UNet++ 모델
  - Segmentation Models PyTorch (SMP) 라이브러리 기반
  - 다양한 인코더 지원 (HRNet, ResNet 등)

- **`unet3plus.py`**: `UNet_3Plus_dsp` - UNet 3+ 모델
  - Deep Supervision 지원
  - 멀티 스케일 특징 추출

- **`segformer.py`**: `Segformer` - SegFormer 모델
  - Transformer 기반 세그멘테이션 모델
  - 효율적인 self-attention 메커니즘

### 모델 관리

- **`model_picker.py`**: `ModelPicker` - 모델 선택 및 생성 관리
  - 설정 파일에서 모델 이름을 받아 해당 모델 인스턴스 생성
  - 모델별 파라미터 전달 및 초기화

## 사용 방법

### 설정 파일에서 사용

```yaml
# HRNet 사용
model_name: HRNet
model_parameter:
  config_path: configs/mmseg_config_py_files/hrnet_w18_fcn.py
  num_classes: 29

# UnetPlusPlus 사용
model_name: UnetPlusPlus
model_parameter:
  encoder_name: tu-hrnet_w18
  encoder_weights: None
  in_channels: 3
  classes: 29
```

### 코드에서 직접 사용

```python
from models.model_picker import ModelPicker

model_selector = ModelPicker()

# HRNet 모델 생성
model = model_selector.get_model(
    "HRNet",
    config_path="configs/mmseg_config_py_files/hrnet_w18_fcn.py",
    num_classes=29
)

# UnetPlusPlus 모델 생성
model = model_selector.get_model(
    "UnetPlusPlus",
    encoder_name="tu-hrnet_w18",
    encoder_weights=None,
    in_channels=3,
    classes=29
)
```

## 모델별 특징

### HRNet
- **장점**: 고해상도 특징 유지, 정확한 세그멘테이션
- **사용 라이브러리**: mmsegmentation
- **디코더 옵션**: FCNHead, OCRHead
- **권장 용도**: 정확도가 중요한 경우

### UnetPlusPlus
- **장점**: 다양한 인코더와 조합 가능, 유연한 구조
- **사용 라이브러리**: Segmentation Models PyTorch (SMP)
- **인코더 옵션**: HRNet, ResNet, EfficientNet 등
- **권장 용도**: 다양한 실험 및 비교

### UNet 3+
- **장점**: Deep Supervision, 멀티 스케일 특징
- **사용 라이브러리**: 커스텀 구현
- **특징**: 보조 출력을 통한 학습 안정화
- **권장 용도**: 복잡한 세그멘테이션 작업

### SegFormer
- **장점**: Transformer 기반, 글로벌 컨텍스트 이해
- **사용 라이브러리**: 커스텀 구현
- **특징**: Self-attention 메커니즘
- **권장 용도**: 대규모 데이터셋, 복잡한 패턴 인식

## 주의사항

- HRNet은 mmsegmentation 라이브러리가 설치되어 있어야 합니다.
- UnetPlusPlus는 Segmentation Models PyTorch (SMP) 라이브러리가 필요합니다.
- 각 모델의 입력 크기와 출력 크기는 모델 구현을 확인하세요.
- 모델 파라미터는 설정 파일의 `model_parameter` 섹션에서 정의됩니다.

