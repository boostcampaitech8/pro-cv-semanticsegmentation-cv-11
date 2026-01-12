# Loss 폴더

세그멘테이션 모델 학습에 사용되는 Loss 함수들을 정의하는 폴더입니다.

## 폴더 구조

```
loss/
├── bce.py                  # Binary Cross Entropy Loss
├── dice.py                  # Dice Loss
├── jaccard.py               # Jaccard (IoU) Loss
├── focal.py                 # Focal Loss
├── tversky.py               # Tversky Loss
├── focal_tversky.py         # Focal Tversky Loss
├── mixed.py                 # Mixed Loss (여러 Loss 조합)
├── loss_mixer.py            # Loss 함수 선택 및 조합 관리
└── README-loss.md           # 이 파일
```

## 주요 파일 설명

### 개별 Loss 함수

- **`bce.py`**: `MyBCELoss` - Binary Cross Entropy Loss
  - 픽셀 단위 분류를 위한 기본 Loss
  - 안정적인 학습에 유리

- **`dice.py`**: `MyDiceLoss` - Dice Loss
  - 세그멘테이션 성능 향상에 효과적
  - 클래스 불균형 문제 완화

- **`jaccard.py`**: `MyJaccardLoss` - Jaccard (IoU) Loss
  - Intersection over Union 기반 Loss
  - Dice Loss와 유사한 특성

- **`focal.py`**: `MyFocalLoss` - Focal Loss
  - 어려운 샘플에 더 집중
  - 클래스 불균형 문제 해결에 효과적

- **`tversky.py`**: `MyTverskyLoss` - Tversky Loss
  - False Positive와 False Negative에 가중치 조절 가능
  - Precision과 Recall 균형 조절

- **`focal_tversky.py`**: `MyFocalTverskyLoss` - Focal Tversky Loss
  - Focal Loss와 Tversky Loss의 조합
  - 어려운 샘플과 클래스 불균형 모두 고려

### Loss 조합 및 관리

- **`mixed.py`**: `MixLoss` - 여러 Loss 함수를 가중 평균으로 조합
  - 예: BCE 30% + Dice 70%
  - 3-Stage Loss Switching에서 사용

- **`loss_mixer.py`**: `LossMixer` - Loss 함수 선택 및 생성 관리
  - 설정 파일에서 Loss 이름을 받아 해당 Loss 인스턴스 생성
  - Mixed Loss의 경우 여러 Loss를 조합하여 생성

## 사용 방법

### 설정 파일에서 사용

```yaml
# 단일 Loss 사용
loss_name: DiceLoss
loss_parameter:
  smooth: 1e-5

# Mixed Loss 사용
loss_name: Mixed
loss_parameter:
  losses:
    - name: BCEWithLogitsLoss
      weight: 0.3
      params: {}
    - name: DiceLoss
      weight: 0.7
      params:
        smooth: 1e-5
```

### 코드에서 직접 사용

```python
from loss.loss_mixer import LossMixer

loss_selector = LossMixer()

# 단일 Loss
criterion = loss_selector.get_loss("DiceLoss", smooth=1e-5)

# Mixed Loss
criterion = loss_selector.get_loss("Mixed", losses=[
    {"name": "BCEWithLogitsLoss", "weight": 0.3, "params": {}},
    {"name": "DiceLoss", "weight": 0.7, "params": {"smooth": 1e-5}}
])
```

## Loss 함수 선택 가이드

### 초기 학습
- **BCE Loss**: 안정적인 학습 시작
- **Dice Loss**: 세그멘테이션 성능 향상

### 클래스 불균형 문제
- **Focal Loss**: 어려운 샘플에 집중
- **Focal Tversky Loss**: Focal + Tversky 조합

### Precision/Recall 균형
- **Tversky Loss**: alpha, beta 파라미터로 조절

### 일반적인 조합
- **BCE + Dice**: 가장 일반적인 조합 (예: 0.3:0.7)
- **3-Stage Loss Switching**: 학습 단계에 따라 비율 조정

## 주의사항

- 모든 Loss 함수는 PyTorch의 `nn.Module`을 상속받아 구현되어 있습니다.
- Mixed Loss의 경우 weight의 합이 1이 되도록 정규화됩니다.
- Loss 함수의 파라미터는 각 Loss의 구현 파일에서 확인할 수 있습니다.

