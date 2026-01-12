# 3-Stage Loss Switching이란?

## 개요

3-Stage Loss Switching은 학습 과정에서 모델의 성능에 따라 **동적으로 loss 함수의 비율을 변경**하는 기법입니다. 초기에는 Dice Loss 중심으로 시작하여 세그멘테이션 성능을 빠르게 향상시키고, 모델이 성장함에 따라 점진적으로 Dice Loss의 비중을 더욱 높여 최종 성능을 최적화합니다.

## 왜 사용하나요?

### 문제 상황
- **초기 학습 단계**: Dice Loss만 사용하면 학습이 불안정할 수 있음
- **후기 학습 단계**: BCE Loss만 사용하면 세그멘테이션 성능 향상에 한계가 있음
- **고정된 Loss 비율**: 학습 전 과정에서 동일한 비율을 사용하면 최적의 성능을 달성하기 어려움

### 해결 방법
학습 단계에 따라 loss 비율을 자동으로 조정하여:
1. **초기 (Stage 1)**: Dice 중심으로 빠른 성능 향상 (BCE:Dice = 0.3:0.7)
2. **중기 (Stage 2)**: Dice 비중 증가 (BCE:Dice = 0.2:0.8)
3. **후기 (Stage 3)**: Dice에 거의 집중하여 최종 성능 최적화 (BCE:Dice = 0.1:0.9)

## 동작 원리

### 3단계 구조

```
Stage 1 (초기) → Stage 2 (중기) → Stage 3 (후기)
BCE:Dice       → BCE:Dice       → BCE:Dice
0.3:0.7        → 0.2:0.8        → 0.1:0.9
```

### 전환 조건

각 stage는 다음 조건을 만족해야 다음 stage로 전환됩니다:

1. **Global Dice Threshold**: 전체 평균 dice 점수가 특정 값 이상
2. **Class Dice Thresholds**: 특정 클래스들의 dice 점수가 각각 특정 값 이상
3. **Consecutive Epochs**: 위 조건을 연속으로 N 에폭 동안 만족

### 예시 설정

```yaml
# 기본 loss 설정 (Stage 1)
loss_name: Mixed
loss_parameter:
  losses:
    - name: BCEWithLogitsLoss
      weight: 0.3  # Stage 1: BCE 30%
    - name: DiceLoss
      weight: 0.7  # Stage 1: Dice 70%

# 3-stage loss switching 설정
loss_switch:
  loss_stages:
    stage1:  # 기본 stage (위의 loss_parameter로 설정됨)
      # stage1은 loss_name과 loss_parameter로 설정됨
    
    stage2:  # Stage 1 -> Stage 2 전환
      enabled: true
      lr_multiplier: 1.0  # Stage 전환 시 LR 조정 (1.0 = 조정 안 함)
      conditions:
        global_dice_threshold: 0.9550  # avg_dice >= 0.9550
        class_dice_thresholds:  # 특정 클래스 dice >= threshold
          Pisiform: 0.85
          Trapezoid: 0.88
        consecutive_epochs: 1  # 연속 1 에폭 만족 필요
      loss:
        loss_name: Mixed
        loss_parameter:
          losses:
            - name: BCEWithLogitsLoss
              weight: 0.2  # Stage 2: BCE 20%, Dice 80%
            - name: DiceLoss
              weight: 0.8
    
    stage3:  # Stage 2 -> Stage 3 전환
      enabled: true
      lr_multiplier: 1.0
      conditions:
        global_dice_threshold: 0.9650  # avg_dice >= 0.9650
        class_dice_thresholds:
          Pisiform: 0.89
          Trapezoid: 0.91
        consecutive_epochs: 1
      loss:
        loss_name: Mixed
        loss_parameter:
          losses:
            - name: BCEWithLogitsLoss
              weight: 0.1  # Stage 3: BCE 10%, Dice 90%
            - name: DiceLoss
              weight: 0.9
```

## 학습 과정 예시

실제 학습 과정에서 다음과 같이 동작합니다:

```
Epoch 1-20:  Stage 1 (BCE:0.3, Dice:0.7)
  - avg_dice: 0.94 → 조건 미달 (0.9550 필요)
  - Pisiform: 0.82 → 조건 미달 (0.85 필요)

Epoch 21:    Stage 1 (BCE:0.3, Dice:0.7)
  - avg_dice: 0.956 ✅
  - Pisiform: 0.86 ✅
  - Trapezoid: 0.89 ✅
  - → Stage 2로 전환!

Epoch 22-25: Stage 2 (BCE:0.2, Dice:0.8)
  - avg_dice: 0.962 → 조건 미달 (0.9650 필요)
  - Pisiform: 0.87 → 조건 미달 (0.89 필요)

Epoch 26:    Stage 2 (BCE:0.2, Dice:0.8)
  - avg_dice: 0.967 ✅
  - Pisiform: 0.90 ✅
  - Trapezoid: 0.92 ✅
  - → Stage 3로 전환!

Epoch 27-30: Stage 3 (BCE:0.1, Dice:0.9)
  - Dice 중심으로 최종 성능 향상
  - 최종 avg_dice: 0.972
```

## 주요 특징

### 1. 자동 전환
- 모델 성능이 조건을 만족하면 자동으로 다음 stage로 전환
- `consecutive_epochs` 조건으로 일시적인 성능 향상에 의한 조기 전환 방지

### 2. 클래스별 조건
- 특정 클래스(Pisiform, Trapezoid 등)의 성능을 별도로 모니터링
- 약한 클래스의 성능이 향상될 때까지 대기

### 3. Learning Rate 조정 (선택사항)
- Stage 전환 시 `lr_multiplier`로 학습률 조정 가능
- 예: `lr_multiplier: 0.5` → LR을 절반으로 감소

### 4. Wandb 로깅
- 현재 stage와 전환 시점이 자동으로 로깅됨
- `val/loss_stage`: 현재 stage (1, 2, 3)
- `val/loss_stage_switch_epoch`: stage 전환 에폭

## 사용 방법

### 1. Config 파일 설정

```yaml
# 기본 loss 설정 (Stage 1)
loss_name: Mixed
loss_parameter:
  losses:
    - name: BCEWithLogitsLoss
      weight: 0.7
    - name: DiceLoss
      weight: 0.3

# 3-stage loss switching 설정
loss_switch:
  loss_stages:
    stage2:
      enabled: true
      conditions:
        global_dice_threshold: 0.80
        class_dice_thresholds:
          Pisiform: 0.75
          Trapezoid: 0.75
        consecutive_epochs: 1
      loss:
        loss_name: Mixed
        loss_parameter:
          losses:
            - name: BCEWithLogitsLoss
              weight: 0.4
            - name: DiceLoss
              weight: 0.6
    stage3:
      enabled: true
      conditions:
        global_dice_threshold: 0.90
        class_dice_thresholds:
          Pisiform: 0.85
          Trapezoid: 0.85
        consecutive_epochs: 1
      loss:
        loss_name: Mixed
        loss_parameter:
          losses:
            - name: BCEWithLogitsLoss
              weight: 0.1
            - name: DiceLoss
              weight: 0.9
```

### 2. 학습 실행

일반적인 학습과 동일하게 실행하면 됩니다:

```bash
python train.py --config configs/hrnet_w18_aug_elastic_3_stage.yaml
```

### 3. 모니터링

- **터미널 출력**: Stage 전환 시 자동으로 출력됨
  ```
  Stage 1->2 condition met (1/1 consecutive epochs)
  Stage 1 -> Stage 2 switching triggered at epoch 11
  ```

- **Wandb**: `val/loss_stage` 그래프에서 stage 변화 확인 가능

## 장점

1. **자동화**: 수동으로 loss 비율을 조정할 필요 없음
2. **점진적 최적화**: Dice Loss 비중을 점진적으로 높여 세그멘테이션 성능 최적화
3. **성능 기반 전환**: 모델 성능이 조건을 만족할 때만 전환하여 안정적
4. **유연성**: 조건과 loss 비율을 자유롭게 설정 가능

## 주의사항

1. **조건 설정**: 너무 높은 threshold는 stage 전환이 일어나지 않을 수 있음
2. **Consecutive Epochs**: 너무 높으면 전환이 늦어질 수 있음
3. **LR Multiplier**: Stage 전환 시 LR을 조정하면 학습 패턴이 변경될 수 있음

## 참고

- 구현 위치: `trainer.py`의 `validate()` 메서드
- Config 예시: `configs/configs_example/251230/hrnet_w18_aug_elastic_3_stage.yaml`

