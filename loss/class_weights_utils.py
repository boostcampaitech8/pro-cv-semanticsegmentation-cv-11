"""
Class Weights 공통 유틸리티 함수
모든 loss 함수에서 공통으로 사용하는 class weights 처리 로직
"""
import torch
from typing import Union, Dict, List, Optional

def parse_class_weights(class_weights: Optional[Union[Dict, List, torch.Tensor]]) -> Optional[torch.Tensor]:
    """
    class_weights를 dict/list/tensor에서 torch.Tensor로 변환
    
    Args:
        class_weights: dict (클래스 이름 -> 가중치), list, 또는 torch.Tensor
        
    Returns:
        torch.Tensor 또는 None
    """
    if class_weights is None:
        return None
    
    # OmegaConf DictConfig를 일반 dict로 변환
    try:
        from omegaconf import DictConfig
        if isinstance(class_weights, DictConfig):
            class_weights = dict(class_weights)
    except ImportError:
        pass
    
    if isinstance(class_weights, dict):
        # dict인 경우: dataset.py의 CLASSES 순서에 맞춰 tensor로 변환
        from dataset import CLASSES
        class_weights_list = [class_weights.get(cls, 1.0) for cls in CLASSES]
        return torch.tensor(class_weights_list, dtype=torch.float32)
    elif isinstance(class_weights, list):
        return torch.tensor(class_weights, dtype=torch.float32)
    elif isinstance(class_weights, torch.Tensor):
        return class_weights
    else:
        # 기타 타입은 tensor로 변환 시도
        return torch.tensor(class_weights, dtype=torch.float32)

def apply_class_weights(
    loss_per_class: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    클래스별 loss에 class weights를 적용
    
    Args:
        loss_per_class: (C,) 형태의 클래스별 loss
        class_weights: (C,) 형태의 클래스별 가중치 tensor
        device: loss_per_class의 device (자동 감지 가능)
        
    Returns:
        가중치가 적용된 loss_per_class
    """
    if class_weights is None:
        return loss_per_class
    
    if not isinstance(class_weights, torch.Tensor):
        # tensor가 아닌 경우 변환 시도
        class_weights = parse_class_weights(class_weights)
        if class_weights is None:
            return loss_per_class
    
    # device 맞추기
    if device is None:
        device = loss_per_class.device
    
    if class_weights.device != device:
        class_weights = class_weights.to(device)
    
    return loss_per_class * class_weights

