from .unetplusplus import UnetPlusPlus
from .unet3plus import UNet_3Plus_dsp
from .segformer import Segformer

class ModelPicker():
    def __init__(self) -> None:
        self.model_classes = {
            "UnetPlusPlus": UnetPlusPlus,
            "Unet3PlusDeepSup": UNet_3Plus_dsp,
            "SegFormer": Segformer
        }

    def get_model(self, model_name, **model_parameter):
        return self.model_classes.get(model_name, None)(**model_parameter)