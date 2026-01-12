# python -m models.comparison

from .unetplusplus import UnetPlusPlus
from .unet3plus import UNet_3Plus_dsp, UNet_3Plus, UNet_3Plus_dsp_ck
from .segformer import Segformer
from .hrnet import HRNet

class ModelPicker():
    def __init__(self) -> None:
        self.model_classes = {
            "UnetPlusPlus": UnetPlusPlus,
            "Unet3Plus": UNet_3Plus,
            "Unet3PlusDeepSup": UNet_3Plus_dsp,
            "Unet3PlusDeepSupCheck": UNet_3Plus_dsp_ck,
            "SegFormer": Segformer,
            "HRNet": HRNet
        }

    def get_model(self, model_name, **model_parameter):
        return self.model_classes.get(model_name, None)(**model_parameter)
    
if __name__ == '__main__':
    # model1 = UnetPlusPlus(n_classes=29)
    # model2 = UNet_3Plus(n_classes=29)
    # model3 = UNet_3Plus_dsp(n_classes=29)
    # model4 = UNet_3Plus_dsp_ck(n_classes=29)
    # model5 = Segformer(n_classes=29)
    # model6 = HRNet(n_classes=29)
    
    # 학습 가능한 파라미터 개수
    trainable_parameters = sum(p.numel() for p in model5.parameters() if p.requires_grad)
    # trainable_parameters = sum(p.numel() for p in model6.parameters() if p.requires_grad)
    
    # 전체 파라미터 개수
    total_parameters = sum(p.numel() for p in model5.parameters())
    # total_parameters = sum(p.numel() for p in model6.parameters())

    print("trainable_parameters : ", trainable_parameters)
    print("total_parameters : ", total_parameters)

# UNet++
# trainable_parameters :  26078609
# total_parameters :  26078609

# UNet3+
# trainable_parameters :  27048477
# total_parameters :  27048477

# UNet3+_deepsup, +ck
# trainable_parameters :  27566417
# total_parameters :  27566417

# SegFormer
# trainable_parameters :  21876545
# total_parameters :  21876545

# HRNet
# trainable_parameters :  9643559
# total_parameters :  9643559


