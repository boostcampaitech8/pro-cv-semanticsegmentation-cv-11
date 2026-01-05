#!/usr/bin/env python3
"""
state_dict만 저장된 모델의 encoder_name을 확인하는 스크립트
"""
import torch
import sys
import os.path as osp

def check_model_config(model_path):
    """
    state_dict의 키를 분석하여 encoder_name을 추론
    """
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # state_dict 추출
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        print("This is a full model, not state_dict.")
        return
    
    # state_dict 키 확인
    keys = list(state_dict.keys())
    print(f"\nTotal keys: {len(keys)}")
    print(f"\nFirst 20 keys:")
    for i, key in enumerate(keys[:20]):
        print(f"  {i+1}. {key}")
    
    # encoder 관련 키 찾기
    encoder_keys = [k for k in keys if 'encoder' in k.lower()]
    print(f"\nEncoder-related keys (first 10):")
    for i, key in enumerate(encoder_keys[:10]):
        print(f"  {i+1}. {key}")
    
    # encoder_name 추론
    print(f"\n=== Encoder Name 추론 ===")
    
    # SE (Squeeze-and-Excitation) 모듈 확인
    se_keys = [k for k in keys if 'se_module' in k.lower() or 'se_block' in k.lower() or 'se.' in k.lower()]
    has_se = len(se_keys) > 0
    
    # Layer 구조 확인
    layer_keys = {}
    for key in keys:
        if 'encoder.layer' in key:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'layer' and i + 1 < len(parts):
                    layer_num = parts[i + 1]
                    if layer_num not in layer_keys:
                        layer_keys[layer_num] = []
                    layer_keys[layer_num].append(key)
    
    print(f"Detected layers: {sorted(layer_keys.keys())}")
    
    # 각 layer의 블록 개수 확인
    layer_block_counts = {}
    for layer_num, layer_key_list in layer_keys.items():
        blocks = set()
        for key in layer_key_list:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == layer_num and i + 1 < len(parts):
                    try:
                        block_num = int(parts[i + 1])
                        blocks.add(block_num)
                    except:
                        pass
        layer_block_counts[layer_num] = len(blocks)
        print(f"  Layer {layer_num}: {len(blocks)} blocks")
    
    # ResNeXt 특성 확인 (grouped convolution)
    resnext_keys = [k for k in keys if 'conv2' in k and 'groups' not in k]  # grouped conv는 보통 다른 패턴
    grouped_conv_keys = [k for k in keys if 'grouped' in k.lower() or ('conv' in k and any('layer' in k and 'conv2' in k for k in keys))]
    
    # SE-ResNeXt vs ResNeXt 구분
    if has_se:
        print(f"  ✓ SE (Squeeze-and-Excitation) 모듈 발견: {len(se_keys)} keys")
        if se_keys:
            print(f"    예시: {se_keys[0]}")
    
    # Layer 개수와 블록 개수로 encoder 추론
    # se_resnext101_32x4d: layer1=3, layer2=4, layer3=23, layer4=3 blocks
    # se_resnext50_32x4d: layer1=3, layer2=4, layer3=6, layer4=3 blocks
    # resnext101_32x8d: layer1=3, layer2=4, layer3=23, layer4=3 blocks
    # resnext50_32x4d: layer1=3, layer2=4, layer3=6, layer4=3 blocks
    
    layer3_blocks = layer_block_counts.get('layer3', 0)
    layer4_blocks = layer_block_counts.get('layer4', 0)
    
    print(f"\n  Layer3 blocks: {layer3_blocks}")
    print(f"  Layer4 blocks: {layer4_blocks}")
    
    # 추론 로직
    found_encoder = None
    if has_se:
        if layer3_blocks >= 20:  # 23 blocks
            found_encoder = 'se_resnext101_32x4d'
        elif layer3_blocks >= 5:  # 6 blocks
            found_encoder = 'se_resnext50_32x4d'
    else:
        if layer3_blocks >= 20:  # 23 blocks
            found_encoder = 'resnext101_32x8d'
        elif layer3_blocks >= 5:  # 6 blocks
            found_encoder = 'resnext50_32x4d'
    
    if found_encoder:
        most_likely = found_encoder
        print(f"\n✓ 추론된 encoder: {most_likely}")
        
        # 모델 구조 추론
        # UnetPlusPlus인지 SegFormer인지 확인
        if any('decoder' in k.lower() or 'segmentation_head' in k.lower() for k in keys):
            if any('transformer' in k.lower() or 'mit' in k.lower() for k in keys):
                model_name = "SegFormer"
            else:
                model_name = "UnetPlusPlus"
        else:
            model_name = "Unknown"
        
        print(f"\n추론된 모델 구조:")
        print(f"  model_name: {model_name}")
        print(f"  encoder_name: {most_likely}")
        print(f"  encoder_weights: null (state_dict만 저장됨)")
        print(f"  in_channels: 3 (일반적으로)")
        print(f"  classes: 29 (일반적으로)")
        
        # JSON 형식으로 출력
        print(f"\n=== JSON 형식 (model_configs.json에 사용) ===")
        config = {
            "model_name": model_name,
            "encoder_name": most_likely,
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 29
        }
        import json
        print(json.dumps(config, indent=2))
    else:
        print("\n⚠ Encoder를 정확히 추론할 수 없습니다.")
        print("\n추가 정보:")
        print(f"  - SE 모듈: {'있음' if has_se else '없음'}")
        print(f"  - Layer 구조: {sorted(layer_keys.keys())}")
        print(f"  - Layer별 블록 수: {layer_block_counts}")
        print("\n수동으로 확인이 필요합니다. 다음 encoder들을 시도해보세요:")
        if has_se:
            print("  - se_resnext101_32x4d")
            print("  - se_resnext50_32x4d")
        else:
            print("  - resnext101_32x8d")
            print("  - resnext50_32x4d")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model_config.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    if not osp.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    check_model_config(model_path)

