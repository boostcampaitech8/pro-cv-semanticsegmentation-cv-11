# kfold_split_gender.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter

# -----------------------------
# 1. 이미지/라벨 목록 준비
# -----------------------------
IMAGE_ROOT = "/data/ephemeral/home/dataset/train/DCM"
LABEL_ROOT = "/data/ephemeral/home/dataset/train/outputs_json"

# 이미지 .png 파일
pngs = sorted([
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
])

# 라벨 .json 파일
jsons = sorted([
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
])

jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}
assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

# -----------------------------
# 2. 성별 정보 읽기
# -----------------------------
meta_df = pd.read_excel("/data/ephemeral/home/dataset/meta_data.xlsx")
# 성별 문자열 정리: "_x0008_여" -> "여", "_x0008_남" -> "남"
meta_df['성별'] = meta_df['성별'].str.replace('_x0008_', '', regex=False)
# ID -> 성별 mapping
id2gender = {f"ID{row['ID']:03d}": row['성별'] for _, row in meta_df.iterrows()}

# -----------------------------
# 3. 그룹 및 성별 레이블 매핑
# -----------------------------
groups = [os.path.dirname(fname) for fname in pngs]  # 같은 인물은 같은 그룹
# 각 그룹(ID)에 대한 성별 레이블
y = np.array([id2gender[g] for g in groups])

# dummy X (필요하지만 실제 값은 안씀)
X = np.ones((len(y), 1))

# -----------------------------
# 4. StratifiedGroupKFold split
# -----------------------------
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
    print(f"\n===== Fold {fold+1} =====")
    
    train_genders = y[train_idx]
    val_genders = y[val_idx]

    # 그룹 겹침 확인
    train_groups = np.array(groups)[train_idx]
    val_groups = np.array(groups)[val_idx]
    overlap = set(train_groups) & set(val_groups)
    if len(overlap) == 0:
        print("  그룹 겹침 없음. 클래스 비율 분포는: ")
    else:
        print(f"  그룹 겹침 발견. -> {overlap}")
    
    # 성별 분포 확인
    print("  Train gender distribution:", dict(Counter(train_genders)))
    print("  Val   gender distribution:", dict(Counter(val_genders)))
