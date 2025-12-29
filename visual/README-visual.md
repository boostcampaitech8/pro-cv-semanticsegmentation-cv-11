# Visual 폴더

이 폴더는 모델 추론 결과를 시각화하고 분석하는 스크립트들을 관리합니다.

## 폴더 구조

```
visual/
├── visual.py                 # 기본 시각화 스크립트
├── visual_with_GT.py         # Ground Truth 포함 시각화
├── visual_run.sh             # 기본 시각화 실행 스크립트
├── visual_run_with_GT.sh     # GT 포함 시각화 실행 스크립트
└── vis_results/              # 시각화 결과 저장 폴더 (gitignore)
```

## 주요 파일 설명

### 시각화 스크립트

- **`visual.py`**: 기본 시각화 스크립트
  - 모델 추론 결과를 이미지로 시각화
  - 원본 이미지, 예측 마스크, 오버레이 등을 생성
  - RLE 인코딩된 마스크를 디코딩하여 시각화

- **`visual_with_GT.py`**: Ground Truth 포함 시각화
  - 예측 결과와 함께 실제 Ground Truth도 함께 표시
  - 비교 분석에 유용
  - 정확도 평가 및 오류 분석에 활용

### 실행 스크립트

- **`visual_run.sh`**: 기본 시각화 실행
  - `visual.py`를 실행하는 wrapper 스크립트
  - 기본 파라미터로 시각화 수행

- **`visual_run_with_GT.sh`**: GT 포함 시각화 실행
  - `visual_with_GT.py`를 실행하는 wrapper 스크립트
  - Ground Truth와 함께 시각화 수행

## 사용법

### 기본 시각화

```bash
cd visual
bash visual_run.sh
```

또는 직접 Python 스크립트 실행:

```bash
python visual/visual.py \
    --image_root /path/to/images \
    --csv_path /path/to/predictions.csv \
    --output_dir visual/vis_results
```

### Ground Truth 포함 시각화

```bash
cd visual
bash visual_run_with_GT.sh
```

또는 직접 Python 스크립트 실행:

```bash
python visual/visual_with_GT.py \
    --image_root /path/to/images \
    --label_root /path/to/labels \
    --csv_path /path/to/predictions.csv \
    --output_dir visual/vis_results
```

## 시각화 결과

시각화 결과는 `vis_results/` 폴더에 저장됩니다:

```
vis_results/
└── {experiment_name}/
    └── {image_id}/
        ├── original.png      # 원본 이미지
        ├── prediction.png    # 예측 마스크
        ├── overlay.png       # 오버레이
        └── {class_name}.png  # 클래스별 마스크
```

### 결과 이미지 종류

1. **원본 이미지**: 입력된 원본 의료 이미지
2. **예측 마스크**: 모델이 예측한 세그멘테이션 결과
3. **오버레이**: 원본 이미지 위에 예측 마스크를 투명하게 오버레이
4. **클래스별 마스크**: 각 클래스(뼈)별로 개별 마스크 이미지

### GT 포함 시각화 결과

GT 포함 시각화의 경우 추가로 다음 이미지들이 생성됩니다:

- **ground_truth.png**: 실제 정답 마스크
- **comparison.png**: 예측과 GT를 나란히 비교
- **error_map.png**: 오류 영역 하이라이트

## 스크립트 파라미터

### visual.py

- `--image_root`: 원본 이미지가 있는 루트 디렉토리
- `--csv_path`: 추론 결과 CSV 파일 경로 (RLE 인코딩 포함)
- `--output_dir`: 시각화 결과 저장 디렉토리
- `--image_size`: 이미지 크기 (기본값: 1024)
- `--class_names`: 클래스 이름 리스트 (선택사항)

### visual_with_GT.py

- `--image_root`: 원본 이미지가 있는 루트 디렉토리
- `--label_root`: Ground Truth JSON 파일이 있는 루트 디렉토리
- `--csv_path`: 추론 결과 CSV 파일 경로
- `--output_dir`: 시각화 결과 저장 디렉토리
- `--image_size`: 이미지 크기 (기본값: 1024)

## RLE 디코딩

시각화 스크립트는 CSV 파일의 RLE(Run-Length Encoding) 형식 마스크를 자동으로 디코딩합니다.

```python
def decode_rle_to_mask(rle, height, width):
    """RLE 문자열을 마스크 이미지로 변환"""
    # 구현 내용은 visual.py 참고
```

## 주의사항

- `vis_results/` 폴더는 gitignore에 포함되어 있어 커밋되지 않습니다.
- 대용량 이미지 시각화 시 디스크 공간 확인 필요
- 시각화 결과는 실험별로 폴더를 분리하여 관리하는 것을 권장합니다.

## 활용 예시

### 모델 성능 분석

1. 학습된 모델로 추론 실행
2. `visual_with_GT.py`로 예측과 GT 비교
3. 오류 패턴 분석 및 모델 개선 방향 도출

### 결과 제출 전 검증

1. 최종 추론 결과 CSV 생성
2. `visual.py`로 샘플 이미지 시각화
3. 비정상적인 예측 확인 및 수정

