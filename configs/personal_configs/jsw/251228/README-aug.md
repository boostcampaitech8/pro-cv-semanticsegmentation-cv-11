# Augmentation 실험 가이드

## 📋 개요

의료용 손 뼈 세그멘테이션 데이터셋의 특성을 고려하여 최적화된 Data Augmentation 전략을 실험합니다.

### 데이터셋 특성
- **엑스레이 이미지**: 밝기 편차가 큼
- **촬영 각도**: 대부분 손바닥을 팔과 일직선으로 촬영하지만, 일부는 손목을 꺾어서 촬영
- **어려운 케이스**: 손목 부분의 겹치는 뼈 두 개를 구별하기 어려움

---

## 🎯 실험 목표

1. **밝기 편차 대응**: 엑스레이 이미지의 밝기 차이를 모델이 잘 처리하도록 학습
2. **각도 변화 모델링**: 손목 각도 변화에 대한 일반화 성능 향상
3. **겹치는 뼈 구별 개선**: 손목 부분의 겹치는 뼈를 더 정확하게 구별

---

## 📁 실험 Config 파일

### 1. `hrnet_w18_aug_basic.yaml` (기본 조합)
가장 안전하고 효과적인 기본 augmentation 조합입니다.

### 2. `hrnet_w18_aug_elastic.yaml` (ElasticTransform 추가)
겹치는 뼈 구별 개선을 위해 ElasticTransform을 추가한 버전입니다.

### 3. `hrnet_w18_aug_strong.yaml` (강한 augmentation)
모든 augmentation을 조합하고 변형 범위를 키운 강한 버전입니다.

---

## 🔧 사용된 Augmentation 상세 설명

### 1. Resize

**무엇인가?**
- 이미지를 지정된 크기로 리사이즈하는 기본 전처리

**인자 설명**
- `width: 1024`, `height: 1024`: 목표 이미지 크기

**적용 근거**
- 모델 입력 크기를 통일하여 일관된 학습 환경 제공
- 메모리 효율과 학습 속도 고려

---

### 2. HorizontalFlip (좌우 반전)

**무엇인가?**
- 이미지를 좌우로 반전시키는 augmentation
- 50% 확률로 적용

**인자 설명**
- `p: 0.5`: 적용 확률 50%

**적용 근거**
- 손은 좌우 대칭 구조이므로 자연스러운 augmentation
- 데이터 증강 효과가 크며 의료 이미지에서 일반적으로 사용
- 추가 비용 없이 데이터를 2배로 늘리는 효과

---

### 3. ShiftScaleRotate (이동/스케일/회전)

**무엇인가?**
- 이미지를 이동(Shift), 스케일(Scale), 회전(Rotate)하는 geometric augmentation
- 세 가지 변형을 동시에 적용

**인자 설명**
- `shift_limit: 0.1` (basic/elastic) / `0.15` (strong)
  - 이미지 크기의 10% 또는 15% 범위 내에서 픽셀 이동
  - 예: 1024x1024 이미지에서 ±102.4픽셀 또는 ±153.6픽셀 이동
- `scale_limit: 0.1` (basic/elastic) / `0.15` (strong)
  - 이미지를 90%~110% 또는 85%~115% 범위로 확대/축소
- `rotate_limit: 15` (basic/elastic) / `20` (strong)
  - 이미지를 ±15도 또는 ±20도 범위로 회전
- `p: 0.5`: 적용 확률 50%

**적용 근거**
- **손목 각도 변화 모델링**: 실제 데이터에서 손목을 꺾어서 촬영한 케이스를 학습에 반영
- **촬영 위치 차이**: 촬영 시 손의 위치나 거리가 달라지는 경우를 모델링
- **작은 각도 사용**: 손은 특정 방향이 있으므로 과도한 회전(예: 90도)은 비현실적
  - 15~20도 범위는 실제 촬영 각도 변화를 잘 반영

---

### 4. RandomBrightnessContrast (밝기/대비 조정)

**무엇인가?**
- 이미지의 밝기와 대비를 랜덤하게 조정하는 augmentation

**인자 설명**
- `brightness_limit: 0.2` (basic/elastic) / `0.3` (strong)
  - 밝기를 ±20% 또는 ±30% 범위로 조정
  - 예: 0.2면 80%~120% 밝기 변화
- `contrast_limit: 0.2` (basic/elastic) / `0.3` (strong)
  - 대비를 ±20% 또는 ±30% 범위로 조정
- `p: 0.5`: 적용 확률 50%

**적용 근거**
- **엑스레이 밝기 편차 대응**: 실제 데이터에서 밝기 편차가 크다는 특성을 반영
- **촬영 조건 차이**: X-ray 장비나 촬영 설정에 따른 밝기 차이를 모델링
- **일반적인 범위**: 의료 이미지에서 ±20~30% 범위가 일반적으로 사용됨

---

### 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)

**무엇인가?**
- 지역적 히스토그램 균등화를 통해 이미지 대비를 개선하는 augmentation
- 이미지를 작은 타일로 나눠 각 타일에서 히스토그램 균등화를 적용

**인자 설명**
- `clip_limit: 2.0` (basic/elastic) / `3.0` (strong)
  - 히스토그램 클리핑 제한값
  - 높을수록 대비가 더 강하게 개선되지만 노이즈 증가 가능
  - 2.0~3.0 범위가 일반적으로 사용됨
- `tile_grid_size: [8, 8]`
  - 이미지를 8x8 타일로 분할하여 각 타일에서 CLAHE 적용
  - 타일이 작을수록 지역적 대비 개선이 세밀하지만 계산 비용 증가
- `p: 0.5`: 적용 확률 50%

**적용 근거**
- **엑스레이 대비 개선**: 엑스레이 이미지의 대비를 개선하여 뼈 구조를 더 명확하게 만듦
- **의료 이미지 표준 기법**: 의료 이미지 전처리에서 널리 사용되는 기법
- **지역적 처리**: 전체 이미지가 아닌 지역적으로 처리하여 자연스러운 결과 생성

---

### 6. ElasticTransform (탄성 변형)

**무엇인가?**
- 이미지를 탄성 변형시켜 픽셀 위치를 부드럽게 변형하는 augmentation
- 겹치는 객체의 경계를 더 명확하게 구별하는 데 도움

**인자 설명**
- `alpha: 50`
  - 변형 강도 (높을수록 변형이 강함)
  - 50은 중간 정도의 변형 강도
- `sigma: 5`
  - 스무딩 정도 (높을수록 부드러운 변형)
  - 5는 적당한 스무딩 정도
- `alpha_affine: 10`
  - 어파인 변형 강도 (전체적인 기하학적 변형)
  - 10은 작은 어파인 변형
- `p: 0.3`: 적용 확률 30% (다른 augmentation보다 낮음)

**적용 근거**
- **겹치는 뼈 구별 개선**: 손목 부분의 겹치는 뼈 두 개를 구별하는 데 도움
- **경계 강조**: 의료 이미지에서 경계를 강조하는 데 효과적
- **보수적 적용**: 과도한 변형을 방지하기 위해 `p=0.3`으로 설정
  - ElasticTransform은 강력한 augmentation이므로 과도하게 사용하면 비현실적인 이미지 생성 가능

---

### 7. RandomGamma (감마 보정)

**무엇인가?**
- 감마 보정을 통해 이미지의 밝기 곡선을 조정하는 augmentation
- 엑스레이 이미지의 밝기 편차를 추가로 모델링

**인자 설명**
- `gamma_limit: [80, 120]`
  - 감마 값 범위 (80~120%, 1.0 기준)
  - 예: 80이면 감마=0.8 (더 밝게), 120이면 감마=1.2 (더 어둡게)
- `p: 0.3`: 적용 확률 30%

**적용 근거**
- **밝기 편차 추가 대응**: RandomBrightnessContrast와 함께 엑스레이 밝기 편차를 더 다양하게 모델링
- **보수적 범위**: 80~120% 범위로 설정하여 비현실적인 밝기 변화 방지
- **strong 버전에만 사용**: 과도한 변형 가능성이 있어 `aug_strong`에만 포함

---

## 📊 Config별 Augmentation 비교

| Augmentation | Basic | Elastic | Strong |
|-------------|-------|---------|--------|
| **HorizontalFlip** | ✅ (p=0.5) | ✅ (p=0.5) | ✅ (p=0.5) |
| **ShiftScaleRotate** | ✅ (limit=0.1, rotate=15°) | ✅ (limit=0.1, rotate=15°) | ✅ (limit=0.15, rotate=20°) |
| **RandomBrightnessContrast** | ✅ (limit=0.2) | ✅ (limit=0.2) | ✅ (limit=0.3) |
| **CLAHE** | ✅ (clip=2.0) | ✅ (clip=2.0) | ✅ (clip=3.0) |
| **ElasticTransform** | ❌ | ✅ (p=0.3) | ✅ (p=0.3) |
| **RandomGamma** | ❌ | ❌ | ✅ (p=0.3) |

---

## 🚀 실험 실행 방법

### 순차 실행
```bash
./scripts/train_251228.sh
```

### 개별 실행
```bash
# 기본 조합
python train.py --config configs/251228/hrnet_w18_aug_basic.yaml

# ElasticTransform 추가
python train.py --config configs/251228/hrnet_w18_aug_elastic.yaml

# 강한 augmentation
python train.py --config configs/251228/hrnet_w18_aug_strong.yaml
```

---

## 📈 추천 실험 순서

1. **`aug_basic`** 먼저 실험
   - 가장 안전하고 효과적인 조합
   - 성능 향상이 확인되면 다음 단계로 진행

2. **`aug_elastic`** 시도
   - 기본 조합에서 성능 향상이 있으면 ElasticTransform 추가
   - 겹치는 뼈 구별 개선 효과 확인

3. **`aug_strong`** 최종 실험
   - 더 강한 augmentation이 필요할 때 시도
   - 과도한 augmentation으로 인한 성능 저하 주의

---

## 💡 Tips

- **로그 확인**: 각 실험의 로그는 `logs/251228/` 폴더에 저장됩니다
- **Wandb 모니터링**: 실험 진행 상황은 Wandb에서 실시간으로 확인 가능
- **Checkpoint 저장**: 각 실험의 best model은 `checkpoints/HRNet_W18/` 폴더에 저장됩니다
- **성능 비교**: Wandb 대시보드에서 세 실험의 성능을 비교하여 최적의 augmentation 조합 선택

---

## 📝 참고사항

- 모든 augmentation은 **학습 시에만 적용**되며, validation과 inference에는 적용되지 않습니다
- `p` 값은 각 augmentation이 적용될 확률을 의미합니다
- 여러 augmentation이 동시에 적용될 수 있으며, 적용 순서는 config 파일에 정의된 순서대로입니다

---

## 🎯 기대 효과

- **일반화 성능 향상**: 다양한 촬영 조건에 대한 모델의 일반화 능력 향상
- **밝기 편차 대응**: 엑스레이 밝기 편차에 대한 강건성 향상
- **각도 변화 대응**: 손목 각도 변화에 대한 일반화 성능 향상
- **겹치는 뼈 구별**: 손목 부분의 겹치는 뼈를 더 정확하게 구별

---

**작성일**: 2024-12-28  
**작성자**: AI Assistant  
**목적**: 의료용 손 뼈 세그멘테이션 Augmentation 실험 가이드

