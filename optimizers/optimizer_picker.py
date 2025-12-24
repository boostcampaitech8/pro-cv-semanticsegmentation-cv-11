import torch.optim as torch_optim

def Adam(**optimzier_parameter):
    return torch_optim.Adam(**optimzier_parameter)

def AdamW(**optimzier_parameter):
    return torch_optim.AdamW(**optimzier_parameter)

class OptimizerPicker():
    def __init__(self) -> None:
        self.optimizer_classes = {
            "adam" : Adam,
            "adamw" : AdamW
        }

    def get_optimizer(self, optimizer_name , **optimizer_parameter):
        return self.optimizer_classes.get(optimizer_name, None)(**optimizer_parameter)