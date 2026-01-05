# 원래 위치: /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/utils/normalization.py
"""
BatchNorm을 GroupNorm으로 교체하는 유틸리티 함수
batch_size=1일 때 gradient accumulation 사용 시 BatchNorm 불안정 문제를 해결하기 위함
"""
import torch.nn as nn


def _find_divisor(num_channels, num_groups):
    """
    num_channels의 약수 중 num_groups에 가장 가까운 값을 찾기
    
    Args:
        num_channels: 채널 수
        num_groups: 원하는 그룹 수
    
    Returns:
        num_channels의 약수 중 num_groups에 가장 가까운 값
    """
    # num_groups가 num_channels보다 크면 num_channels를 반환
    if num_groups >= num_channels:
        return num_channels
    
    # num_channels의 약수 찾기
    divisors = []
    for i in range(1, int(num_channels ** 0.5) + 1):
        if num_channels % i == 0:
            divisors.append(i)
            if i != num_channels // i:
                divisors.append(num_channels // i)
    
    divisors = sorted(divisors, reverse=True)
    
    # num_groups보다 작거나 같으면서 가장 큰 약수 찾기
    for divisor in divisors:
        if divisor <= num_groups:
            return divisor
    
    # 약수를 찾지 못한 경우 (거의 발생하지 않음) 1 반환
    return 1


def replace_bn_with_gn(model, num_groups=32):
    """
    모델의 모든 BatchNorm2d를 GroupNorm으로 교체
    
    Args:
        model: PyTorch 모델
        num_groups: GroupNorm의 그룹 수 (기본값: 32)
                    채널 수의 약수로 자동 조정됨
    
    Returns:
        교체된 모델 (in-place 수정이므로 원본 모델 반환)
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # BatchNorm 파라미터 가져오기
            num_channels = module.num_features
            eps = module.eps
            affine = module.affine
            
            # num_channels의 약수 중 num_groups에 가장 가까운 값 찾기
            actual_groups = _find_divisor(num_channels, num_groups)
            
            # GroupNorm으로 교체
            gn = nn.GroupNorm(
                num_groups=actual_groups,
                num_channels=num_channels,
                eps=eps,
                affine=affine
            )
            
            # 가중치 복사 (affine=True인 경우)
            if affine:
                gn.weight.data = module.weight.data.clone()
                gn.bias.data = module.bias.data.clone()
            
            # 모델에 교체
            setattr(model, name, gn)
        else:
            # 재귀적으로 자식 모듈 처리
            replace_bn_with_gn(module, num_groups)
    
    return model


def count_bn_layers(model):
    """
    모델 내 BatchNorm2d 레이어 개수 세기
    
    Args:
        model: PyTorch 모델
    
    Returns:
        BatchNorm2d 레이어 개수
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            count += 1
    return count

