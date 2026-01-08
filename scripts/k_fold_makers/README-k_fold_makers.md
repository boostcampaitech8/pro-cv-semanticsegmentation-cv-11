# K-Fold Makers 폴더

K-Fold 교차 검증을 위한 데이터 분할 스크립트들입니다.

## 폴더 구조

```
scripts/k_fold_makers/
├── create_kfold_splits.py   # K-Fold 분할 Python 스크립트
├── create_kfold_splits.sh   # K-Fold 분할 실행 스크립트
└── README-k_fold_makers.md   # 이 파일
```

## 주요 파일 설명

### create_kfold_splits.py

K-Fold 데이터 분할을 수행하는 Python 스크립트입니다.

**주요 기능:**
- GroupKFold를 사용한 데이터 분할
- Patient ID를 기준으로 그룹화하여 분할
- train/val split을 파일 시스템에 직접 저장
- 특정 fold만 생성하는 옵션 지원

**사용법:**
```bash
python scripts/k_fold_makers/create_kfold_splits.py \
    --image_root /path/to/train/images \
    --label_root /path/to/train/labels \
    --output_root /path/to/output \
    --n_splits 5 \
    --target_fold 0  # 선택사항: 특정 fold만 생성
```

**주요 인자:**
- `--image_root`: 학습 이미지 루트 경로
- `--label_root`: 학습 라벨 루트 경로
- `--output_root`: 분할 결과 저장 루트 경로
- `--n_splits`: K-Fold 개수 (기본값: 5)
- `--target_fold`: 특정 fold만 생성 (선택사항, 없으면 모든 fold 생성)

**출력 구조:**
```
output_root/
├── fold0/
│   ├── train/
│   │   ├── DCM/
│   │   └── outputs_json/
│   └── val/
│       ├── DCM/
│       └── outputs_json/
├── fold1/
│   └── ...
└── ...
```

### create_kfold_splits.sh

K-Fold 분할을 실행하는 Shell 스크립트입니다.

**주요 특징:**
- 상단에 변수 선언으로 설정 관리
- 특정 fold만 생성하는 옵션 지원

**사용법:**
```bash
# 스크립트 상단의 변수들을 수정
IMAGE_ROOT="/path/to/train/images"
LABEL_ROOT="/path/to/train/labels"
OUTPUT_ROOT="/path/to/output"
N_SPLITS=5
TARGET_FOLD=0  # 또는 빈 값으로 모든 fold 생성

# 실행
./scripts/k_fold_makers/create_kfold_splits.sh
```

## 동작 원리

### GroupKFold 사용

- Patient ID를 기준으로 그룹화
- 같은 환자의 데이터가 train/val에 분리되지 않도록 보장
- 데이터 누수 방지

### 데이터 분할 과정

1. 이미지와 라벨 파일 수집
2. Patient ID 추출 (파일명에서)
3. GroupKFold로 K개 fold 생성
4. 각 fold의 train/val 데이터를 파일 시스템에 복사

### Seed 고정

- `RANDOM_SEED = 21` 사용
- `train.py`와 동일한 seed 값
- 재현 가능한 분할 결과

## 사용 예시

### 모든 Fold 생성

```bash
python scripts/k_fold_makers/create_kfold_splits.py \
    --image_root /data/ephemeral/home/dataset/train/DCM \
    --label_root /data/ephemeral/home/dataset/train/outputs_json \
    --output_root /data/ephemeral/home/dataset \
    --n_splits 5
```

### 특정 Fold만 생성

```bash
python scripts/k_fold_makers/create_kfold_splits.py \
    --image_root /data/ephemeral/home/dataset/train/DCM \
    --label_root /data/ephemeral/home/dataset/train/outputs_json \
    --output_root /data/ephemeral/home/dataset \
    --n_splits 5 \
    --target_fold 0
```

## 주의사항

- 데이터 분할 전에 충분한 디스크 공간을 확보하세요.
- 모든 fold를 생성하면 원본 데이터의 K배 공간이 필요합니다.
- 특정 fold만 생성하면 공간을 절약할 수 있습니다.
- 분할 결과는 `output_root`에 저장되며, 원본 데이터는 변경되지 않습니다.

