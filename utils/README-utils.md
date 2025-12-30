# Utils 폴더

프로젝트 전반에서 사용되는 유틸리티 함수들을 관리하는 폴더입니다.

## 폴더 구조

```
utils/
├── wandb.py                  # Wandb 설정 및 초기화
└── README-utils.md           # 이 파일
```

## 주요 파일 설명

### wandb.py

Weights & Biases (Wandb) 설정 및 초기화를 담당하는 모듈입니다.

**주요 기능:**
- Wandb API 키 로드 (`.env` 파일에서)
- Wandb run 초기화
- 실험 설정 로깅

**사용법:**
```python
from utils.wandb import set_wandb

# 설정 파일에서 Wandb 초기화
wandb_run = set_wandb(configs)
```

**설정 파일에서 필요한 값:**
- `team_name`: Wandb 팀 이름
- `project_name`: Wandb 프로젝트 이름
- `experiment_detail`: 실험 이름
- `model_name`: 모델 이름
- `image_size`: 이미지 크기
- `train_batch_size`: 배치 크기
- `loss_name`: Loss 함수 이름
- `scheduler_name`: Scheduler 이름
- `lr`: 학습률
- `max_epoch`: 최대 에폭
- `optimizer_name`: Optimizer 이름

**환경 변수 설정:**

`.env` 파일에 Wandb API 키를 설정해야 합니다:

```
WANDB_API_KEY=your_api_key_here
```

## 주의사항

- Wandb API 키는 `.env` 파일에 저장되며, `.gitignore`에 포함되어 있어 Git에 커밋되지 않습니다.
- `load_dotenv()`를 통해 환경 변수를 로드합니다.
- Wandb를 사용하지 않는 경우 이 모듈은 사용되지 않습니다.

