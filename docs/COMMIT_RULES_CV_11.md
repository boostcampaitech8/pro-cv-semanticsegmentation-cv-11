# Github commit, issue, PR guide & rules docs.

##  CV-11 팀의 커밋 룰입니다.

## 기본적으로는, 이슈에 내용 작성 후 커밋에선 이슈 태깅을 권장.

### 1. 커밋

- `git commit` 하고 엔터 누르면 vi 에디터 열려서 commit 메시지 여러 줄로 적을 수 있음.
    - 최소 한 줄, 본인 희망에 따라 여러 줄 가능

```bash
# 기본 사용법:
- type: 간단한 설명

# type 종류:
feat: 새로운 기능 추가
refactor: 코드 리팩토링
docs: 문서 작성/수정
fix: 버그 수정
chore: 진짜 자잘한 것들, 그다지 안 중요한 커밋들(잡일)

# 예시:
feat: add HRNet model support from mmsegmentation
fix: resolve inference resize issue
refactor: group wandb metrics with train/val prefixes

# issue 태깅:
해결한 issue가 있을 경우, #(이슈 넘버)로 태깅해도 좋음.
fix: #8 - add typecasting code to float32 before data aug in dataset.py
```

### 2. 이슈

```bash
# 기본 사용법:
- [Type] 간단한 설명

# Type 종류:
- [Feature]: 새로운 기능 추가
- [Refactor]: 코드 리팩토링
- [Docs]: 문서 작성/수정
- [Fix]: 버그 수정
- [Task]: 작업/태스크 -> 범용적으로 사용 가능

# 예시
[Refactor] .yaml 파일에 checkpoint_name_format을 추가: .pt 파일의 이름을 지정 가능하게 함
[Feature] HRNet support from mmsegmentation
[Fix] inference.py에서 resize 문제 해결
```

### 3. PR

```bash
# 기본 사용법:
- (바로) 간단한 설명

# 예시:
- yaml 파일에 checkpoint.pt 이름을 지정 가능하게 함 + 학습 자동화 scripts 초안 작성
- HRNet 기본 baseline 작성 + wandb logging 리더보드 Refactoring

# PR 생성 전 확인사항:
- 관련 Issue가 있다면 Description에 언급
  - PR 이후 해당 Issue close하기
- 관련 Issue가 없다면 간단하게 글로 설명 적기
- (필요시) Reviewer에 인원 추가해 코드 리뷰 요청
```