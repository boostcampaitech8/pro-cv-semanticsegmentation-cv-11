from .unetplusplus import UnetPlusPlus
from .unet3plus import UNet_3Plus_dsp, UNet_3Plus, UNet_3Plus_dsp_ck
from .unet3plusGN import UNet_3Plus_GN
from .segformer import Segformer
from .hrnet import HRNet

class ModelPicker():
    def __init__(self) -> None:
        self.model_classes = {
            "UnetPlusPlus": UnetPlusPlus,
            "Unet3Plus": UNet_3Plus,
            "Unet3PlusGN": UNet_3Plus_GN,
            "Unet3PlusDeepSup": UNet_3Plus_dsp,
            "Unet3PlusDeepSupCheck": UNet_3Plus_dsp_ck,
            "SegFormer": Segformer,
            "HRNet": HRNet
        }

    def get_model(self, model_name, **model_parameter):
        return self.model_classes.get(model_name, None)(**model_parameter)