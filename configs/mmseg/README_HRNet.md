# HRNet 모델 사용 가이드

이 프로젝트에 HRNet 모델을 추가했습니다. HRNet은 mmsegmentation 라이브러리에서 제공하는 모델로, 기존의 SMP(Segmentation Models PyTorch) 기반 모델들과 동일한 방식으로 사용할 수 있습니다.

## 사용법법

HRNet을 사용하기 위해서는 mmsegmentation과 mmcv가 필요합니다:

```bash
# mmcv 설치
pip install -U openmim
mim install mmengine
# mim install "mmcv>=2.0.0" -> 에러 발생, https://github.com/open-mmlab/mmcv/issues/3096 대신
mim install "mmcv==2.1.0"
(mmsegmentation 폴더는 참고 용으로 clone한 거고 저기서 install하진 않을 거임 무시)
(적은 것 외의 의존성 문제 발생할수도 있음 흠...)

# mmsegmentation 설치
pip install "mmsegmentation>=1.0.0"

# 이후 의존성 문제 대량 발생...
# 아래 코드들 입력해서 직접 requirements.txt 입력하니 해결은 됨.
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -r requirements.txt
cd ..

# 넘파이가 자꾸 2.0 이상으로 깔려서 다시 1.x로 설치해줘야 정상 작동함.
pip install 'numpy<2.0'

# train
python train.py --config configs/hrnet_w18_config.yaml

# inference
python inference.py /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/checkpoints/HRNet_W18/best_2epoch_0.7430.pt
    --image_root /data/ephemeral/home/dataset/test/DCM \
    --thr 0.5 \
    --output /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/outputs/HRNet_W18/hrnet_fcnhead_251226.csv \
    --resize 2048
```

## 모델 구조

HRNet은 `models/hrnet.py`에 구현되어 있으며, mmsegmentation의 HRNet을 래핑하여 기존 프로젝트 구조와 호환되도록 만들었습니다.

### 주요 특징

- **SMP 모델과 동일한 인터페이스**: `forward()` 메서드가 동일한 입력/출력 형식을 사용합니다
- **Config 파일 지원**: mmsegmentation의 config 파일을 직접 사용할 수 있습니다
- **유연한 설정**: config 파일 없이도 기본 설정으로 모델을 생성할 수 있습니다

## 사용 방법

### 1. Config 파일을 사용한 학습

#### HRNet-W18 사용

```bash
python train.py --config configs/hrnet_w18_config.yaml
```

#### HRNet-W48 사용

```bash
python train.py --config configs/hrnet_w48_config.yaml
```

### 2. Config 파일 구조

HRNet config 파일은 다른 모델들과 동일한 구조를 따릅니다:

```yaml
model_name: HRNet
model_parameter:
  config_path: configs/mmseg/hrnet_w18_fcn.py  # 또는 hrnet_w48_fcn.py
  num_classes: 29
  # pretrained: 'open-mmlab://msra/hrnetv2_w18'  # 선택사항
```

### 3. 모델 파라미터

- `config_path` (str, optional): mmsegmentation config 파일 경로. None이면 기본 HRNet-W18 설정 사용
- `num_classes` (int): 출력 클래스 수. 기본값: 29
- `pretrained` (str, optional): 사전 학습된 가중치 경로 또는 'open-mmlab://msra/hrnetv2_w18' / 'open-mmlab://msra/hrnetv2_w48'

### 4. Config 파일 위치

mmsegmentation config 파일들은 `configs/mmseg/` 폴더에 있습니다:

- `configs/mmseg/hrnet_w18_fcn.py`: HRNet-W18 FCN 설정
- `configs/mmseg/hrnet_w48_fcn.py`: HRNet-W48 FCN 설정

이 파일들은 mmsegmentation의 원본 config를 기반으로 하되, 프로젝트의 데이터 전처리 파이프라인과 호환되도록 수정되었습니다.

## 모델 비교

| 모델 | 파라미터 수 | 메모리 사용량 | 특징 |
|------|------------|--------------|------|
| HRNet-W18 | 적음 | 낮음 (~2.9GB) | 빠른 학습, 적은 메모리 |
| HRNet-W48 | 많음 | 높음 (~6.2GB) | 더 높은 성능, 많은 메모리 필요 |

**권장 배치 사이즈:**
- HRNet-W18: `train_batch_size: 2`, `val_batch_size: 4`
- HRNet-W48: `train_batch_size: 1`, `val_batch_size: 2`

## 주의사항

1. **메모리 사용량**: HRNet-W48은 메모리를 많이 사용하므로, GPU 메모리가 부족하면 배치 사이즈를 줄이거나 이미지 크기를 조정하세요.

2. **데이터 전처리**: HRNet은 mmsegmentation의 기본 전처리(`data_preprocessor`)를 사용하지 않도록 설정되어 있습니다. 대신 프로젝트의 `dataset.py`에서 처리하는 전처리를 사용합니다.

3. **Forward 메서드**: HRNet의 `forward()` 메서드는 SMP 모델들과 동일하게 `(B, C, H, W)` 형태의 텐서를 입력받고 `(B, num_classes, H, W)` 형태의 logits를 반환합니다.

## 문제 해결

### ImportError: No module named 'mmseg'

mmsegmentation이 설치되지 않았습니다. 위의 설치 요구사항을 참고하여 설치하세요.

### CUDA out of memory

배치 사이즈를 줄이거나 이미지 크기를 줄여보세요. HRNet-W48의 경우 특히 메모리를 많이 사용합니다.

### Config 파일을 찾을 수 없음

`config_path`가 올바른지 확인하세요. 상대 경로는 프로젝트 루트를 기준으로 합니다.

## 참고 자료

- [mmsegmentation 공식 문서](https://mmsegmentation.readthedocs.io/)
- [HRNet 논문](https://arxiv.org/abs/1908.07919)
- [mmsegmentation HRNet configs](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/hrnet)

