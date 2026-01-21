## 빠른 시작
0. 먼저, root/data에 현재 대회의 data폴더로 구성되어있어야 합니다.
1. Repository Clone
먼저 메인 저장소와 MMSegmentation 라이브러리를 클론합니다.
```bash
# 메인 저장소 클론
git clone https://github.com/boostcampaitech8/pro-cv-semanticsegmentation-cv-11.git
cd pro-cv-semanticsegmentation-cv-11
# MMSegmentation 클론 (프로젝트 내부)
git clone https://github.com/open-mmlab/mmsegmentation.git
```
2. Environment Setup
가상환경을 생성하고 필요한 라이브러리를 설치합니다.
```bash
# 1. venv 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate
# 2. 필수 라이브러리 설치
pip install --upgrade pip
pip install -r requirements.txt
```
3. Environment Variable (PYTHONPATH)
MMSegmentation 모듈을 인식할 수 있도록 파이썬 경로를 설정합니다.
```bash
# 환경 변수 설정 (현재 세션 적용)
export PYTHONPATH=$(pwd)/mmsegmentation/:$PYTHONPATH
# (선택) 터미널 접속 시 자동 적용을 원할 경우
echo "export PYTHONPATH=$(pwd)/mmsegmentation/:\$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
```
4. Training
준비된 설정을 통해 모델 학습을 실행합니다.
```bash
# 학습 실행
python train.py --config configs/review_unetpp_2048_2_2_aug_fp16.yaml
# 혹은 shell script도 있다.
./train.sh
```
