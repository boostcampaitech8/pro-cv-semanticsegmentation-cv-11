# HRNet-W18 FCN config for mmsegmentation
# Based on mmsegmentation/configs/_base_/models/fcn_hr18.py

### 출처 ###
# /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/mmsegmentation/configs/_base_/models/fcn_hr18.py
# 에서, 나머진 다 그대로 쓰되, 
# data_preprocessor만 data_preprocessor = None으로 변경한 것.
############

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
# Disable data_preprocessor to avoid conflicts with existing preprocessing
# (we handle preprocessing in our dataset pipeline)
data_preprocessor = None

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
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
        channels=sum([18, 36, 72, 144]),  # 270
        input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=29,  # Will be overridden by model_parameter.num_classes
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

