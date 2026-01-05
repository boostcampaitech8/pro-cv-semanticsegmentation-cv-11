"""
GroupNorm 교체 함수 검증 테스트
"""
import torch
import torch.nn as nn
import sys
import os

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.normalization import replace_bn_with_gn, count_bn_layers
from models.model_picker import ModelPicker


def test_simple_model():
    """간단한 모델로 테스트"""
    print("=" * 60)
    print("Test 1: Simple Model with BatchNorm")
    print("=" * 60)
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.seq = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.seq(x)
            return x
    
    model = SimpleModel()
    print(f"Before: {count_bn_layers(model)} BatchNorm layers")
    
    # BatchNorm 확인
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            print(f"  - Found BatchNorm: {name}")
    
    # GroupNorm으로 교체
    model = replace_bn_with_gn(model, num_groups=32)
    print(f"\nAfter: {count_bn_layers(model)} BatchNorm layers")
    
    # GroupNorm 확인
    gn_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm):
            gn_count += 1
            print(f"  - Found GroupNorm: {name} (groups={module.num_groups}, channels={module.num_channels})")
    
    print(f"Total GroupNorm layers: {gn_count}")
    
    # Forward pass 테스트
    x = torch.randn(2, 3, 32, 32)
    try:
        out = model(x)
        print(f"✓ Forward pass successful: {out.shape}")
        assert count_bn_layers(model) == 0, "Some BatchNorm layers remain!"
        assert gn_count == 3, f"Expected 3 GroupNorm layers, got {gn_count}"
        print("✓ Test 1 PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}\n")
        return False


def test_hrnet():
    """HRNet 모델로 테스트"""
    print("=" * 60)
    print("Test 2: HRNet Model (mmseg)")
    print("=" * 60)
    
    try:
        model_picker = ModelPicker()
        # HRNet은 num_classes를 받음
        model = model_picker.get_model("HRNet", num_classes=29)
        
        bn_count_before = count_bn_layers(model)
        print(f"Before: {bn_count_before} BatchNorm layers")
        
        if bn_count_before == 0:
            print("No BatchNorm layers found (model may use other normalization)")
            return True
        
        # GroupNorm으로 교체
        model = replace_bn_with_gn(model, num_groups=32)
        bn_count_after = count_bn_layers(model)
        
        # GroupNorm 개수 확인
        gn_count = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
        
        print(f"After: {bn_count_after} BatchNorm layers")
        print(f"GroupNorm layers: {gn_count}")
        
        # Forward pass 테스트
        x = torch.randn(1, 3, 256, 256)
        try:
            out = model(x)
            print(f"✓ Forward pass successful: {out.shape}")
            assert bn_count_after == 0, f"Expected 0 BatchNorm layers, got {bn_count_after}"
            assert gn_count == bn_count_before, f"Expected {bn_count_before} GroupNorm layers, got {gn_count}"
            print("✓ Test 2 PASSED\n")
            return True
        except Exception as e:
            print(f"✗ Forward pass failed: {e}\n")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_unetplusplus():
    """UnetPlusPlus 모델로 테스트"""
    print("=" * 60)
    print("Test 3: UnetPlusPlus Model")
    print("=" * 60)
    
    try:
        model_picker = ModelPicker()
        # SMP의 UnetPlusPlus는 encoder_name과 classes를 받음
        model = model_picker.get_model("UnetPlusPlus", 
                                       encoder_name="resnet34",
                                       encoder_weights="imagenet",
                                       classes=29)
        
        bn_count_before = count_bn_layers(model)
        print(f"Before: {bn_count_before} BatchNorm layers")
        
        if bn_count_before == 0:
            print("No BatchNorm layers found (model may use other normalization)")
            return True
        
        # GroupNorm으로 교체
        model = replace_bn_with_gn(model, num_groups=32)
        bn_count_after = count_bn_layers(model)
        
        # GroupNorm 개수 확인
        gn_count = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
        
        print(f"After: {bn_count_after} BatchNorm layers")
        print(f"GroupNorm layers: {gn_count}")
        
        # Forward pass 테스트
        x = torch.randn(1, 3, 256, 256)
        try:
            out = model(x)
            print(f"✓ Forward pass successful: {out.shape}")
            assert bn_count_after == 0, f"Expected 0 BatchNorm layers, got {bn_count_after}"
            assert gn_count == bn_count_before, f"Expected {bn_count_before} GroupNorm layers, got {gn_count}"
            print("✓ Test 3 PASSED\n")
            return True
        except Exception as e:
            print(f"✗ Forward pass failed: {e}\n")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_segformer():
    """SegFormer 모델로 테스트"""
    print("=" * 60)
    print("Test 4: SegFormer Model")
    print("=" * 60)
    
    try:
        model_picker = ModelPicker()
        # SegFormer는 encoder_name과 classes를 받음
        model = model_picker.get_model("SegFormer",
                                      encoder_name="mit_b3",
                                      encoder_weights="imagenet",
                                      classes=29)
        
        bn_count_before = count_bn_layers(model)
        print(f"Before: {bn_count_before} BatchNorm layers")
        
        # SegFormer는 LayerNorm을 사용하므로 BatchNorm이 없을 수 있음
        if bn_count_before == 0:
            print("No BatchNorm layers found (SegFormer uses LayerNorm, not BatchNorm)")
            print("This is expected behavior for SegFormer")
            return True
        
        # GroupNorm으로 교체
        model = replace_bn_with_gn(model, num_groups=32)
        bn_count_after = count_bn_layers(model)
        
        # GroupNorm 개수 확인
        gn_count = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
        
        print(f"After: {bn_count_after} BatchNorm layers")
        print(f"GroupNorm layers: {gn_count}")
        
        # Forward pass 테스트
        x = torch.randn(1, 3, 256, 256)
        try:
            out = model(x)
            print(f"✓ Forward pass successful: {out.shape}")
            assert bn_count_after == 0, f"Expected 0 BatchNorm layers, got {bn_count_after}"
            if bn_count_before > 0:
                assert gn_count == bn_count_before, f"Expected {bn_count_before} GroupNorm layers, got {gn_count}"
            print("✓ Test 4 PASSED\n")
            return True
        except Exception as e:
            print(f"✗ Forward pass failed: {e}\n")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_nested_modules():
    """중첩된 모듈 테스트"""
    print("=" * 60)
    print("Test 5: Nested Modules")
    print("=" * 60)
    
    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(3, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.block2 = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(64, 128, 3),
                    nn.BatchNorm2d(128)
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, 3),
                    nn.BatchNorm2d(256)
                )
            ])
        
        def forward(self, x):
            x = self.block1(x)
            for block in self.block2:
                x = block(x)
            return x
    
    model = NestedModel()
    bn_count_before = count_bn_layers(model)
    print(f"Before: {bn_count_before} BatchNorm layers")
    
    model = replace_bn_with_gn(model, num_groups=32)
    bn_count_after = count_bn_layers(model)
    gn_count = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
    
    print(f"After: {bn_count_after} BatchNorm layers")
    print(f"GroupNorm layers: {gn_count}")
    
    x = torch.randn(1, 3, 32, 32)
    try:
        out = model(x)
        print(f"✓ Forward pass successful")
        assert bn_count_after == 0, "Some BatchNorm layers remain!"
        assert gn_count == 3, f"Expected 3 GroupNorm layers, got {gn_count}"
        print("✓ Test 3 PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}\n")
        return False


def test_channel_limits():
    """채널 수가 그룹 수보다 작은 경우 테스트"""
    print("=" * 60)
    print("Test 6: Channel Limits (channels < num_groups)")
    print("=" * 60)
    
    class SmallChannelModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(8)   # 8 channels, num_groups=32
            self.bn2 = nn.BatchNorm2d(16)  # 16 channels, num_groups=32
    
    model = SmallChannelModel()
    model = replace_bn_with_gn(model, num_groups=32)
    
    gn_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm):
            gn_count += 1
            actual_groups = module.num_groups
            channels = module.num_channels
            print(f"  - {name}: {channels} channels, {actual_groups} groups")
            assert actual_groups <= channels, f"Groups ({actual_groups}) > channels ({channels})!"
    
    print(f"✓ All GroupNorm layers have groups <= channels")
    print("✓ Test 4 PASSED\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GroupNorm Replacement Function Test")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("Simple Model", test_simple_model()))
    results.append(("HRNet", test_hrnet()))
    results.append(("UnetPlusPlus", test_unetplusplus()))
    results.append(("SegFormer", test_segformer()))
    results.append(("Nested Modules", test_nested_modules()))
    results.append(("Channel Limits", test_channel_limits()))
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if all(result for _, result in results):
        print("✓ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED!")
        sys.exit(1)

