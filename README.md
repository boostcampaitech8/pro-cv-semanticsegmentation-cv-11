### README.md last modified: 260120

<div align="center">
  <h2>🦴 Hand Bone Image Segmentation</h2>

  <div style="display: flex; justify-content: center; gap: 20px;">
    <img
      width="328"
      height="505"
      alt="Image1"
      src="https://github.com/user-attachments/assets/37991d89-db4e-474b-879f-454aa1af7bf4"
    />
    <img
      width="328"
      height="505"
      alt="Image2"
      src="https://github.com/user-attachments/assets/9713b6cb-82ff-4cea-b6bc-30ebbe21ebd2"
    />
  </div>
</div>


## Project Overview
뼈는 인체의 구조와 기능을 담당하는 핵심 요소로, **정확한 Bone Segmentation은 의료 영상 기반 질병 진단과 치료·수술 계획 수립에 필수적**이다.

딥러닝 기반 뼈 Segmentation 기술은 골절·변형 분석, 의료기기 제작, 의료 교육 등 **다양한 의료 분야에서 중요한 역할**을 수행한다.

- **Competition Period** : 2025.12.17 ~ 2026.01.06
- **Input**
  - 손 뼈 X-ray 이미지
  - Segmentation annotation은 json 파일 형태로 제공됨.
- **Output**
  - 모델은 **29개 클래스 각각에 대한 확률 맵**(multi-channel output)을 예측
  - 각 픽셀을 가장 높은 확률을 갖는 클래스로 할당하여 segmentation 결과 생성
  - 최종 예측 결과를 Run-Length Encoding (RLE) 형식으로 변환
  - 제출 형식에 맞게 csv 파일로 저장하여 제출

## Dataset

- Total images: 1,088 (train: 800, test: 288)
- Image size: 2048 x 2048
- Classes: 29
  - 손가락 뼈(finger-1~finger-19)
  - 손목 및 팔 뼈('Trapezium', 'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate', 'Triquetrum', 'Pisiform', 'Radius', 'Ulna')
- Annotation:
  - Train 데이터에 한해 segmentation annotation 제공
  - Run-Length Encoding (RLE) 형식으로 제공
  - 이미지 이름과 클래스별 RLE 정보가 포함
- Additional metadata: 각 이미지에 대해 나이, 성별, 키, 체중 정보를 포함한 meta_data.xlsx 파일이 추가로 제공


## Solution Overview
<p align="center">
  <img src="https://github.com/user-attachments/assets/90a8e69d-456d-44f9-b08d-7fc7269fe2a1" width="90%">
</p>

- Team Notion, Waight&Biases, Google Sheets를 통한 실험 관리 및 실시간 공유를 통한 협업 진행

## Project Result
<h3 align="center">
  <a href="https://mature-shark-e53.notion.site/Hand-Bone-Image-Segmentation-Wrap-up-Report-2e3474a0526a8175bcc0c35766f04037?source=copy_link">📄 Wrap-up Report Link</a>
</h3>
<p align="center"><strong>Public Leaderboard (3/13)</strong></p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/57f918b8-bc48-4af9-a46d-23c2bafd4d80" alt="Public Leaderboard" width="80%">
</p>

<br>

<p align="center"><strong>Private Leaderboard (1/13)🥇</strong></p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/2a9a73ed-e11a-4806-b257-eabe29c75a0c" alt="Private Leaderboard" width="80%">
</p>

## Team Members

<table align="center">
    <tr align="center">
        <td><img src="https://github.com/user-attachments/assets/63c982d2-cc44-474c-9b73-c142627df75e" width="140"></td>
        <td><img src="https://github.com/user-attachments/assets/5c459428-9ffa-4506-b59d-a880a63413b9" width="140"></td>
        <td><img src="https://github.com/user-attachments/assets/ffd16ff0-3c70-4cd1-9f29-f9ce3beda107" width="140"></td>
        <td><img src="https://github.com/user-attachments/assets/9f4be4be-083c-4ce7-948b-6c1e57ed3ed9" width="140"></td>
        <td><img src="https://github.com/user-attachments/assets/a5fc0ec6-1645-4e2e-a4bd-a249b0f9c87a" width="140"></td>
    </tr>
    <tr align="center">
        <td><a href="https://github.com/hyeongseokgo" target="_blank">고형석</a></td>
        <td><a href="https://github.com/M1niJ" target="_blank">김민진</a></td>
        <td><a href="https://github.com/uss0302-cmd" target="_blank">류제윤</a></td>
        <td><a href="https://github.com/Ea3124" target="_blank">이승재</a></td>
        <td><a href="https://github.com/cuffyluv" target="_blank">주상우</a></td>
    </tr>
    <tr align="center">
        <td>T8012</td>
        <td>T8028</td>
        <td>T8065</td>
        <td>T8155</td>
        <td>T8199</td>
    </tr>
</table>

## Role

| Member | Roles |
|--------|-------|
| **고형석** | WandB 세팅, Baseline Refactoring and Modularization, Segformer 모델 실험 |
| **김민진** | EDA 및 데이터 전처리, Augmentation test, DeepLabV3, FCNhead, HRNet 모델 실험 |
| **류제윤** | EDA 및 데이터 전처리, Augmentation, wrist crop test, UNet++ - efficient, Resnet, DenseNet 모델 실험 |
| **이승재** | Baseline Refactoring and Modularization, Input image size, Hyperparameter, Batch size test, UNet3+ , UNet++ 모델 실험  |
| **주상우** | Class-wise and Hyperparameter 실험, Experiment 정리, Ensemble, and TTA Evaluation, UNet++, FCNhead, HRNet 모델 실험   |
---

## File Structure
```
pro-cv-semanticsegmentation-cv-11/
│
├── train.py                  # 학습 메인 스크립트
├── train.sh                  # 학습 실행 스크립트
├── inference.py              # 추론 메인 스크립트
├── inference.sh              # 추론 실행 스크립트
├── trainer.py                # 학습 로직 구현
├── dataset.py                # 데이터셋 정의
│
├── configs/                  # 설정 파일들
│   ├── *.yaml                # 모델별 학습 설정
│   ├── mmseg_config_py_files/  # mmsegmentation 설정
│   ├── class_thresholds/      # 클래스별 threshold 설정
│   ├── class_weights/         # 클래스별 가중치 설정
│   └── personal_configs/      # 개인별 설정 파일들
│
├── models/                   # 모델 아키텍처 정의
│   ├── hrnet.py              # HRNet 모델
│   ├── unetplusplus.py       # UNet++ 모델
│   └── model_picker.py       # 모델 선택 관리
│
├── loss/                     # Loss 함수들
│   ├── bce.py                # Binary Cross Entropy
│   ├── dice.py               # Dice Loss
│   └── loss_mixer.py         # Loss 조합 관리
│
├── scripts/                  # 보조 스크립트들
│   ├── ensemble/             # 앙상블 관련
│   ├── k_fold_makers/        # K-Fold 데이터 분할
│   ├── visualizer/           # 결과 시각화
│   └── custom_runners/        # 커스텀 실행 스크립트들
│
├── optimizers/               # Optimizer 정의
├── scheduler/                # Learning Rate Scheduler 정의
├── utils/                    # 유틸리티 함수들
├── docs/                     # 문서 파일들
└── legacy_files/             # 레거시 파일들 (참고용)
```

### 주요 폴더 설명

- **configs/** : 모델, 데이터, 학습 설정을 YAML로 관리  
  → [configs/README-configs.md](configs/README-configs.md)

- **models/** : HRNet, UNet++, SegFormer 등 세그멘테이션 모델 정의  
  → [models/README-models.md](models/README-models.md)

- **loss/** : BCE, Dice, Focal 등 Loss 함수 및 조합 구현  
  → [loss/README-loss.md](loss/README-loss.md)

- **scripts/** : 앙상블, 데이터 분할, 시각화 등 보조 스크립트  
  → [scripts/README-scripts.md](scripts/README-scripts.md)


## Workflow Summary
적을예정
## Reports & Presentation
- **CV-11 Wrap-up Reports** :
- **CV-11 Presentation** : 
- **Team Notion** :

---
## 빠른 시작

### 1. 학습 (Training)

설정 파일을 작성한 후 학습을 시작합니다:

```bash
# 1. 모델 config 파일 수정
vi configs/hrnet_w18_config.yaml

# 2. 학습 스크립트 수정
vi train.sh

# 3. 학습 실행
./train.sh
```

또는 직접 Python 스크립트 실행:

```bash
python train.py --config configs/hrnet_w18_config.yaml
```

### 2. 추론 (Inference)

학습된 모델로 테스트 이미지에 대한 예측을 수행합니다:

```bash
# 1. 추론 스크립트 수정
vi inference.sh

# 2. 추론 실행
./inference.sh
```

### 3. 앙상블

여러 모델의 예측 결과를 결합합니다:

```bash
# 1. 앙상블 스크립트 수정
vi scripts/ensemble/ensemble.sh

# 2. 앙상블 실행
./scripts/ensemble/ensemble.sh
```

## 요구사항

- Python 3.8+
- PyTorch
- mmsegmentation (mmseg 기반 HRNet 사용 시)
- Segmentation Models PyTorch (UNet++ 사용 시)
- 기타 필수 라이브러리 (requirements.txt 참고)

