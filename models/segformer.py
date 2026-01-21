import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
    
class Segformer(nn.Module):
    def __init__(self,
                 **kwargs):
        super(Segformer, self).__init__()
        self.model = smp.Segformer(**kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(x)