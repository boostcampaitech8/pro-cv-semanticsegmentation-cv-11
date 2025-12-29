"""
K-fold 분할을 파일 시스템에 저장하는 스크립트
- k-fold-checker.ipynb의 1, 2, 3번째 셀 방식 사용 (GroupKFold)
- k-fold-maker.ipynb의 저장 코드 사용
"""
import os
import os.path as osp
import shutil
import argparse
import random
import torch
from sklearn.model_selection import GroupKFold
import numpy as np

# train.py와 동일한 seed 값
RANDOM_SEED = 21


def set_seed(seed):
    """train.py와 동일한 set_seed 함수"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def collect_files(image_root, label_root):
    """
    k-fold-checker.ipynb Cell 0 (1번째 셀)과 동일한 방식으로 파일 수집
    """
    # baseline_FCN.py와 동일한 방식으로 파일 찾기
    pngs = sorted([
        osp.relpath(osp.join(root, fname), start=image_root)
        for root, _dirs, files in os.walk(image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    ])

    jsons = sorted([
        osp.relpath(osp.join(root, fname), start=label_root)
        for root, _dirs, files in os.walk(label_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".json"
    ])

    # baseline_FCN.py와 동일한 검증
    jsons_fn_prefix = {osp.splitext(fname)[0] for fname in jsons}
    pngs_fn_prefix = {osp.splitext(fname)[0] for fname in pngs}
    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

    return np.array(pngs), np.array(jsons)


def perform_groupkfold(pngs, jsons, n_splits=5):
    """
    k-fold-checker.ipynb Cell 1 (2번째 셀)과 동일한 방식으로 GroupKFold 수행
    """
    _filenames = np.array(pngs)
    _labelnames = np.array(jsons)

    # baseline_FCN.py와 동일한 groups 생성
    groups = [osp.dirname(fname) for fname in _filenames]

    # dummy label (baseline_FCN.py와 동일)
    ys = [0 for fname in _filenames]

    # baseline_FCN.py와 동일한 GroupKFold 설정
    # GroupKFold는 기본적으로 shuffle=False, random_state 없음
    gkf = GroupKFold(n_splits=n_splits)

    # 각 fold별 train/val 인덱스 저장
    fold_splits = {}
    for i, (train_idx, val_idx) in enumerate(gkf.split(_filenames, ys, groups)):
        fold_splits[i] = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'train_groups': np.array(groups)[train_idx],
            'val_groups': np.array(groups)[val_idx]
        }
        print(f"Fold {i}: train={len(train_idx)}, val={len(val_idx)}")
        print(f"  Train groups: {len(set(fold_splits[i]['train_groups']))}, Val groups: {len(set(fold_splits[i]['val_groups']))}")
        
        # 겹침 확인
        overlap = set(fold_splits[i]['train_groups']) & set(fold_splits[i]['val_groups'])
        print(f"  Overlap: {'None ✅' if len(overlap) == 0 else overlap}")
        print()
    
    return fold_splits


def create_fold_id_split(fold_splits):
    """
    k-fold-maker.ipynb 방식으로 fold_id_split 생성
    """
    fold_id_split = {}
    
    for fold, split_data in fold_splits.items():
        val_ids = set(split_data['val_groups'])
        train_ids = set(split_data['train_groups'])
        
        fold_id_split[fold] = {
            "train_ids": sorted(train_ids),
            "val_ids": sorted(val_ids)
        }
        
        print(f"Fold {fold}: train={len(train_ids)}, val={len(val_ids)}")
    
    return fold_id_split


def copy_id_folders(id_list, src_root, dst_root):
    """
    k-fold-maker.ipynb의 copy_id_folders 함수
    """
    os.makedirs(dst_root, exist_ok=True)

    for id_ in id_list:
        src_dir = osp.join(src_root, id_)
        dst_dir = osp.join(dst_root, id_)

        if not osp.exists(src_dir):
            print(f"⚠️ Missing source: {src_dir}")
            continue

        if osp.exists(dst_dir):
            # 이미 있으면 skip (중복 방지)
            continue

        shutil.copytree(src_dir, dst_dir)


def save_folds(fold_id_split, image_root, label_root, output_root, target_fold=None):
    """
    k-fold-maker.ipynb의 저장 코드 사용
    """
    if target_fold is not None:
        fold_indices = [target_fold]
        print(f"\n=== Creating fold {target_fold} dataset only ===")
    else:
        fold_indices = list(fold_id_split.keys())
        print(f"\n=== Creating all folds ===")
    
    for fold, split in fold_id_split.items():
        if fold not in fold_indices:
            continue
            
        print(f"\n=== Creating fold {fold} dataset ===")

        fold_dir = osp.join(output_root, f"fold{fold}")

        # train / val 디렉토리
        train_dcm = osp.join(fold_dir, "train", "DCM")
        train_json = osp.join(fold_dir, "train", "outputs_json")
        val_dcm = osp.join(fold_dir, "val", "DCM")
        val_json = osp.join(fold_dir, "val", "outputs_json")

        # copy train
        print(f"  Train DCM 복사 중...")
        copy_id_folders(split["train_ids"], image_root, train_dcm)
        print(f"  Train outputs_json 복사 중...")
        copy_id_folders(split["train_ids"], label_root, train_json)

        # copy val
        print(f"  Val DCM 복사 중...")
        copy_id_folders(split["val_ids"], image_root, val_dcm)
        print(f"  Val outputs_json 복사 중...")
        copy_id_folders(split["val_ids"], label_root, val_json)

        print(f"✅ Fold {fold} done")
        
        # 검증: train/val ID 분리 확인
        train_ids_actual = set(os.listdir(train_dcm)) if osp.exists(train_dcm) else set()
        val_ids_actual = set(os.listdir(val_dcm)) if osp.exists(val_dcm) else set()
        overlap = train_ids_actual & val_ids_actual

        assert len(overlap) == 0, f"❌ Fold {fold} overlap detected: {overlap}"

        print(f"✅ Fold {fold}: train/val ID separation OK "
              f"(train={len(train_ids_actual)}, val={len(val_ids_actual)})")


def main():
    parser = argparse.ArgumentParser(description="K-fold 분할을 파일 시스템에 저장")
    parser.add_argument("--image_root", type=str, required=True,
                        help="원본 이미지 루트 디렉토리")
    parser.add_argument("--label_root", type=str, required=True,
                        help="원본 라벨 루트 디렉토리")
    parser.add_argument("--output_root", type=str, required=True,
                        help="fold 분할 결과를 저장할 루트 디렉토리 (dataset 폴더)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="K-fold 분할 수 (기본값: 5)")
    parser.add_argument("--target_fold", type=int, default=None,
                        help="생성할 fold 번호 (None이면 모든 fold 생성, 예: 0이면 fold0만)")
    
    args = parser.parse_args()
    
    # 0. Seed 설정 (train.py와 동일)
    set_seed(RANDOM_SEED)
    print(f"Seed set to {RANDOM_SEED} (train.py와 동일)")
    
    # 1. k-fold-checker.ipynb Cell 0: 파일 수집
    print("\n" + "=" * 60)
    print("[1/4] 파일 수집 중... (k-fold-checker.ipynb Cell 0)")
    print("=" * 60)
    pngs, jsons = collect_files(args.image_root, args.label_root)
    print(f"Total images: {len(pngs)}")
    print(f"Total labels: {len(jsons)}")
    
    # 2. k-fold-checker.ipynb Cell 1: GroupKFold 수행
    print("\n" + "=" * 60)
    print("[2/4] GroupKFold 수행 중... (k-fold-checker.ipynb Cell 1)")
    print("=" * 60)
    fold_splits = perform_groupkfold(pngs, jsons, args.n_splits)
    
    # 3. k-fold-maker.ipynb 방식으로 fold_id_split 생성
    print("\n" + "=" * 60)
    print("[3/4] fold_id_split 생성 중... (k-fold-maker.ipynb 방식)")
    print("=" * 60)
    fold_id_split = create_fold_id_split(fold_splits)
    
    # 4. k-fold-maker.ipynb 방식으로 저장
    print("\n" + "=" * 60)
    print("[4/4] 파일 시스템에 저장 중... (k-fold-maker.ipynb 저장 코드)")
    print("=" * 60)
    save_folds(fold_id_split, args.image_root, args.label_root, args.output_root, args.target_fold)
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"\n생성된 구조:")
    print(f"  {args.output_root}/")
    fold_indices = [args.target_fold] if args.target_fold is not None else list(range(args.n_splits))
    for fold_idx in fold_indices:
        print(f"    fold{fold_idx}/")
        print(f"      train/")
        print(f"        DCM/")
        print(f"        outputs_json/")
        print(f"      val/")
        print(f"        DCM/")
        print(f"        outputs_json/")


if __name__ == "__main__":
    main()

