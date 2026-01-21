# Scripts 폴더

학습, 추론, 데이터 전처리, 앙상블, 시각화 등의 작업을 자동화하는 스크립트들을 관리하는 폴더입니다.

## 폴더 구조

```
scripts/
├── ensemble/                # 앙상블 관련 스크립트
│   ├── ensemble.py
│   ├── ensemble.sh
│   └── README-ensemble.md
│
├── k_fold_makers/           # K-Fold 교차 검증 데이터 분할
│   ├── create_kfold_splits.py
│   ├── create_kfold_splits.sh
│   └── README-k_fold_makers.md
│
├── visualizer/              # 시각화 관련 스크립트
│   ├── visual.py
│   ├── visual_with_GT.py
│   ├── visual_run.sh
│   └── README-visualizer.md
│
└── README-scripts.md         # 이 파일
```

## 하위 폴더 설명

### ensemble/
여러 모델의 예측 결과를 앙상블하여 최종 결과를 생성하는 스크립트들입니다.
- 1-step 앙상블: 여러 모델을 동시에 로드하여 배치 단위로 앙상블
- 가중 평균 지원
- TTA (Test Time Augmentation) 지원

자세한 내용은 `scripts/ensemble/README-ensemble.md`를 참고하세요.

### k_fold_makers/
K-Fold 교차 검증을 위한 데이터 분할 스크립트들입니다.
- GroupKFold를 사용한 데이터 분할
- train/val split 파일 생성
- 파일 시스템에 직접 저장

자세한 내용은 `scripts/k_fold_makers/README-k_fold_makers.md`를 참고하세요.

### visualizer/
모델 추론 결과를 시각화하는 스크립트들입니다.
- RLE 인코딩된 마스크 디코딩 및 시각화
- Ground Truth와 비교 시각화
- 클래스별 마스크 시각화

자세한 내용은 `scripts/visualizer/README-visualizer.md`를 참고하세요.

## 주의사항

- 각 하위 폴더의 상세한 사용법은 해당 폴더의 README 파일을 참고하세요.
- 스크립트 실행 전 실행 권한 확인: `chmod +x script_name.sh`
- 스크립트 내 경로는 환경에 맞게 수정이 필요할 수 있습니다.
