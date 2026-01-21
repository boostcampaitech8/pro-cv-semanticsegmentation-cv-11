import torch
import torch.nn as nn
from mmengine import Config
from mmseg.models import build_segmentor

class HRNet(nn.Module):
    """
    HRNet wrapper for mmsegmentation.
    
    This class wraps mmsegmentation's HRNet model to be compatible with
    the existing project structure.
    
    Args:
        config_path (str): Path to mmsegmentation config file. 
            If None, uses default HRNet-W18 config.
        num_classes (int): Number of output classes. Default: 29.
        pretrained (str, optional): Path to pretrained weights or 
            'open-mmlab://msra/hrnetv2_w18' for ImageNet pretrained.
        **kwargs: Additional arguments to override config settings.
    """
    def __init__(self, 
                 config_path: str = None,
                 num_classes: int = 29,
                 pretrained: str = None,
                 **kwargs):
        super(HRNet, self).__init__()
        
        if config_path:
            # Load config from file
            cfg = Config.fromfile(config_path)
        else:
            # Create default config
            cfg = self._create_default_config(num_classes, pretrained)
        
        # Override num_classes
        if 'decode_head' in cfg.model:
            decode_head = cfg.model['decode_head']
            # decode_head가 리스트인 경우 (CascadeEncoderDecoder, e.g., OCRNet)
            if isinstance(decode_head, list):
                for head in decode_head:
                    if isinstance(head, dict):
                        head['num_classes'] = num_classes
            # decode_head가 딕셔너리인 경우 (EncoderDecoder, e.g., FCN)
            elif isinstance(decode_head, dict):
                decode_head['num_classes'] = num_classes
        
        # Override pretrained if provided
        if pretrained is not None:
            cfg.model['pretrained'] = pretrained
        
        # Override other settings from kwargs
        for key, value in kwargs.items():
            if key in cfg.model:
                cfg.model[key] = value
            elif 'backbone' in cfg.model and key in cfg.model['backbone']:
                cfg.model['backbone'][key] = value
            elif 'decode_head' in cfg.model:
                decode_head = cfg.model['decode_head']
                # decode_head가 리스트인 경우
                if isinstance(decode_head, list):
                    for head in decode_head:
                        if isinstance(head, dict) and key in head:
                            head[key] = value
                # decode_head가 딕셔너리인 경우
                elif isinstance(decode_head, dict) and key in decode_head:
                    decode_head[key] = value
        
        # Build model
        self.model = build_segmentor(cfg.model)
    
    ### 실제로는 이거 실행 안 됨. if config_path: 에서 걸리기 때문. ###
    ### 즉 이 파일은 인코더 hrnet만 정의하고 있음. ###
    def _create_default_config(self, num_classes, pretrained):
        """Create default HRNet-W18 config."""
        from mmengine import ConfigDict
        
        norm_cfg = dict(type='BN', requires_grad=True)
        
        # Disable data_preprocessor to avoid conflicts with existing preprocessing
        model = dict(
            type='EncoderDecoder',
            data_preprocessor=None,  # Disable mmseg's preprocessing
            pretrained=pretrained or 'open-mmlab://msra/hrnetv2_w18',
            backbone=dict(
                type='HRNet',
                norm_cfg=norm_cfg,
                norm_eval=False,
                extra=dict(
                    stage1=dict(
                        num_modules=1,
                        num_branches=1,
                        block='BOTTLENECK',
                        num_blocks=(4,),
                        num_channels=(64,)),
                    stage2=dict(
                        num_modules=1,
                        num_branches=2,
                        block='BASIC',
                        num_blocks=(4, 4),
                        num_channels=(18, 36)),
                    stage3=dict(
                        num_modules=4,
                        num_branches=3,
                        block='BASIC',
                        num_blocks=(4, 4, 4),
                        num_channels=(18, 36, 72)),
                    stage4=dict(
                        num_modules=3,
                        num_branches=4,
                        block='BASIC',
                        num_blocks=(4, 4, 4, 4),
                        num_channels=(18, 36, 72, 144)))),
            decode_head=dict(
                type='FCNHead',
                in_channels=[18, 36, 72, 144],
                in_index=(0, 1, 2, 3),
                channels=270,  # sum([18, 36, 72, 144])
                input_transform='resize_concat',
                kernel_size=1,
                num_convs=1,
                concat_input=False,
                dropout_ratio=-1,
                num_classes=num_classes,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', 
                    use_sigmoid=False, 
                    loss_weight=1.0)),
            train_cfg=dict(),
            test_cfg=dict(mode='whole'))
        
        cfg = ConfigDict(dict(model=model))
        return cfg
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
                Expected to be normalized to [0, 1] range (already preprocessed)
            
        Returns:
            torch.Tensor: Segmentation logits of shape (B, num_classes, H, W)
        """
        # mmsegmentation's _forward returns seg_logits directly
        # Use mode='tensor' to get raw tensor output without post-processing
        # data_samples=None to avoid requiring SegDataSample objects
        try:
            output = self.model(x, mode='tensor', data_samples=None)
        except Exception:
            # Fallback: directly call _forward if mode doesn't work
            output = self.model._forward(x, data_samples=None)
        
        # _forward returns a tensor (seg_logits)
        # If it's a tuple/list, take the first element
        ### 이거 ocrhead라고 걸리는 거 아님. 그냥 배치로 했을 때 걸림.
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

