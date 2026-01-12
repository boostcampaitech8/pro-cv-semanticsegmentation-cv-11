from torch.optim import lr_scheduler
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

def RLROP(optimizer, **scheduler_parameter):
    return lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_parameter)

def CosAnn(optimizer, **scheduler_parameter):
    return lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_parameter)

def CosWarmup(optimizer, **scheduler_parameter):
    return CosineAnnealingWarmupRestarts(optimizer, **scheduler_parameter)

class SchedulerPicker():
    def __init__(self, optimizer) -> None:
        self.scheduler_classes = {
            "ReduceLROnPlateau" : RLROP,
            "CosineAnnealingLR" : CosAnn,
            "CosineAnnealingWarmupRestarts" : CosWarmup
        }
        self.optimizer = optimizer

    def get_scheduler(self, scheduler_name, **scheduler_parameter):
        return self.scheduler_classes.get(scheduler_name, None)(self.optimizer, **scheduler_parameter)