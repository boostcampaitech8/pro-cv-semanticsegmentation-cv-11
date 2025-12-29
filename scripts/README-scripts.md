# Scripts 폴더

이 폴더는 학습, 추론, 데이터 전처리 등의 작업을 자동화하는 스크립트들을 관리합니다.

## 폴더 구조

```
scripts/
├── automatic_runners/         # 자동화된 학습 실행 스크립트들
│   ├── train_all.sh          # 모든 모델 학습 실행
│   ├── train_easy.sh         # 간단한 학습 실행
│   ├── train_multiple.sh     # 여러 설정 파일 순차 실행
│   └── train_super_easy.sh   # 초간단 학습 실행
├── basic_runners/             # 기본 학습/추론 스크립트들
│   ├── train.sh        # 학습 실행
│   ├── inference.sh    # 추론 실행
│   ├── inference_thr.sh # 추론 (threshold 적용)
│   ├── inference_for_vis.sh # 추론 (시각화용)
│   └── class_thresholds.json # 클래스별 threshold 설정
├── k_fold_makers/            # K-Fold 교차 검증 데이터 분할
│   ├── create_kfold_splits.py # K-Fold 분할 Python 스크립트
│   └── create_kfold_splits.sh # K-Fold 분할 실행 스크립트
```

## 주요 스크립트 설명

### automatic_runners/

자동화된 학습 실행 스크립트들입니다. 여러 모델이나 설정을 한 번에 실행할 때 사용합니다.

- **`train_all.sh`**: 모든 모델 학습을 순차적으로 실행
- **`train_easy.sh`**: 간단한 설정으로 빠른 학습 실행
- **`train_multiple.sh`**: 여러 설정 파일을 순차적으로 실행
- **`train_super_easy.sh`**: 최소 설정으로 테스트용 학습 실행

**사용 예시:**
```bash
cd scripts/automatic_runners
bash train_all.sh
```

### basic_runners/

기본적인 학습 및 추론 작업을 수행하는 스크립트들입니다.

- **`hrnet_train.sh`**: HRNet 모델 학습 실행
  - 기본 설정: `configs/hrnet_w18_config.yaml`
  - 필요시 설정 파일 경로 수정 가능

- **`hrnet_inference.sh`**: HRNet 모델 추론 실행
  - 학습된 모델로 테스트 데이터 추론
  - 결과를 CSV 형식으로 저장

- **`hrnet_inference_thr.sh`**: Threshold 적용 추론
  - 클래스별 threshold를 적용한 추론
  - `class_thresholds.json` 파일 사용

- **`hrnet_inference_for_vis.sh`**: 시각화용 추론
  - 추론 결과를 시각화하기 위한 형식으로 저장
  - `visual/` 폴더의 스크립트와 연동

- **`class_thresholds.json`**: 클래스별 threshold 값 설정
  - 각 클래스에 대한 최적 threshold 값 저장
  - 추론 시 이 값을 사용하여 이진화

**사용 예시:**
```bash
cd scripts/basic_runners
bash hrnet_train.sh
bash hrnet_inference.sh
```

### k_fold_makers/

K-Fold 교차 검증을 위한 데이터 분할 스크립트들입니다.

- **`create_kfold_splits.py`**: K-Fold 데이터 분할 Python 스크립트
  - 데이터를 K개 fold로 분할
  - train/val split 파일 생성

- **`create_kfold_splits.sh`**: K-Fold 분할 실행 스크립트
  - Python 스크립트를 실행하는 wrapper

**사용 예시:**
```bash
cd scripts/k_fold_makers
bash create_kfold_splits.sh
```

## 스크립트 작성 가이드

### 기본 구조

모든 스크립트는 다음 구조를 따르는 것을 권장합니다:

```bash
#!/bin/bash

# 스크립트 설명
# 사용법: bash script_name.sh

# 프로젝트 루트로 이동
cd /data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11

# Python 스크립트 실행
python train.py --config configs/your_config.yaml
```

### 경로 설정

- 절대 경로 사용 권장 (프로젝트 루트 기준)
- 상대 경로 사용 시 `cd` 명령으로 작업 디렉토리 보장

## 주의사항

- `custom_runners/`와 `legacy_EDM/` 폴더는 gitignore에 포함되어 있어 커밋되지 않습니다.
- 스크립트 실행 전 실행 권한 확인: `chmod +x script_name.sh`
- 스크립트 내 경로는 환경에 맞게 수정 필요할 수 있습니다.

