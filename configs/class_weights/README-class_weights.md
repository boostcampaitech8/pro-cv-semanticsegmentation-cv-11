# Class Weights 설정

이 디렉토리에는 클래스별 loss 가중치를 설정하는 JSON 파일들이 있습니다.

## 파일 형식

JSON 파일은 클래스 이름을 키로, 가중치를 값으로 하는 딕셔너리 형식입니다:

```json
{
  "finger-1": 1.0,
  "finger-2": 1.0,
  ...
  "Trapezium": 2.0,
  "Trapezoid": 2.0,
  ...
}
```

## 사용 방법

YAML 설정 파일에서 `class_weights_path`를 지정하면 됩니다:

```yaml
# class weights 설정 (선택적)
class_weights_path: configs/class_weights/class_weights_wrist_2x.json
```

## 제공되는 파일

- **`class_weights_default.json`**: 모든 클래스에 동일한 가중치 (1.0)
- **`class_weights_wrist_2x.json`**: 손목뼈 8개 클래스에 2.0, 나머지 1.0

## 클래스 목록 (29개)

1. finger-1 ~ finger-19 (19개)
2. Trapezium, Trapezoid, Capitate, Hamate, Scaphoid, Lunate, Triquetrum, Pisiform (손목뼈 8개)
3. Radius, Ulna (2개)

## 동작 원리

1. `train.py`에서 JSON 파일을 읽어 `class_weights` 딕셔너리를 생성
2. 각 loss 함수 (DiceLoss, BCELoss 등)에 `class_weights` 파라미터로 전달
3. Loss 계산 시 각 클래스별 loss에 해당 가중치를 곱하여 최종 loss 계산

## 예시

손목뼈에 더 높은 가중치를 주고 싶다면:

```json
{
  "Trapezium": 2.0,
  "Trapezoid": 2.0,
  "Capitate": 2.0,
  "Hamate": 2.0,
  "Scaphoid": 2.0,
  "Lunate": 2.0,
  "Triquetrum": 2.0,
  "Pisiform": 2.0,
  ...
}
```

이렇게 설정하면 손목뼈 클래스의 loss가 2배로 가중치가 적용되어, 모델이 손목뼈 학습에 더 집중하게 됩니다.

