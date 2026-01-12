import torch.optim as torch_optim
from adamp import AdamP

def Adam(**optimzier_parameter):
    return torch_optim.Adam(**optimzier_parameter)

def AdamW(**optimzier_parameter):
    return torch_optim.AdamW(**optimzier_parameter)

def AdamP_optimizer(**optimizer_parameter):
    return AdamP(**optimizer_parameter)

class OptimizerPicker():
    def __init__(self) -> None:
        self.optimizer_classes = {
            "adam" : Adam,
            "adamw" : AdamW,
            "adamp" : AdamP_optimizer
        }

    def get_optimizer(self, optimizer_name , **optimizer_parameter):
        return self.optimizer_classes.get(optimizer_name, None)(**optimizer_parameter)