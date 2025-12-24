import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
    
class UnetPlusPlus(nn.Module):
    def __init__(self,
                 **kwargs):
        super(UnetPlusPlus, self).__init__()
        self.model = smp.UnetPlusPlus(**kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(x)