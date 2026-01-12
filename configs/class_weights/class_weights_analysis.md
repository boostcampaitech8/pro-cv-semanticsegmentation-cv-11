# Class Weight 분석 및 전략

## Dice Score 분석 (낮은 순서)

1. **Pisiform**: 0.9182 ⚠️ (가장 낮음)
2. **Trapezoid**: 0.9271 ⚠️
3. **Trapezium**: 0.9538 ⚠️
4. **Hamate**: 0.9573 ⚠️
5. **Triquetrum**: 0.9595 ⚠️
6. **Lunate**: 0.9655
7. **Capitate**: 0.9670
8. **Scaphoid**: 0.9753

**결론**: 가장 낮은 8개가 **정확히 Carpal Bone 8개와 일치**합니다! ✅

## 평균 비교

- **Carpal Bone 평균**: ~0.955
- **Finger 평균**: ~0.979
- **Arm (Radius, Ulna)**: ~0.990

**Carpal Bone이 약 0.024 (2.4%) 낮음**

## Class Weight 전략 및 파일

### 전략 1: 역비례 가중치 (Inverse Dice Weight) ⚠️
**파일**: `class_weights_inverse_dice.json`

- 가중치 범위: 1.000 ~ 1.079
- **문제점**: 가중치 차이가 너무 작아서 효과가 미미할 수 있음
- **추천도**: ⭐⭐ (너무 보수적)

### 전략 2: 구간별 가중치 (Tier-based) ✅ **추천**
**파일**: `class_weights_tier_based.json`

- **0.90-0.93**: 3.0x (Pisiform, Trapezoid)
- **0.93-0.96**: 2.5x (Trapezium, Hamate, Triquetrum)
- **0.96-0.97**: 2.0x (Lunate, Capitate)
- **0.97-0.98**: 1.5x (Scaphoid, finger-1, finger-16, finger-17)
- **0.98+**: 1.0x (나머지)

**장점**: Dice 점수에 따라 세밀하게 가중치 부여
**추천도**: ⭐⭐⭐⭐⭐

### 전략 3: Carpal Bone 집중 (Simple) ✅ **간단하고 효과적**
**파일**: 
- `class_weights_carpal_25x.json` (2.5x)
- `class_weights_carpal_30x.json` (3.0x)

- **Carpal Bone 8개**: 2.5x 또는 3.0x
- **나머지**: 1.0x

**장점**: 간단하고 명확, Carpal Bone에 집중
**추천도**: ⭐⭐⭐⭐

## 추천 순서

1. **`class_weights_tier_based.json`** - 가장 세밀하고 합리적
2. **`class_weights_carpal_30x.json`** - Carpal Bone에 강하게 집중
3. **`class_weights_carpal_25x.json`** - Carpal Bone에 적당히 집중

