from .bce import MyBCELoss
from .dice import MyDiceLoss
from .jaccard import MyJaccardLoss
from .focal import MyFocalLoss
from .tversky import MyTverskyLoss
from .focal_tversky import MyFocalTverskyLoss
from .mixed import MixLoss

class LossMixer():
    def __init__(self) -> None:
        self.loss_classes = {
            "BCEWithLogitsLoss" : MyBCELoss,
            "DiceLoss": MyDiceLoss,
            "JaccardLoss": MyJaccardLoss,
            "FocalLoss": MyFocalLoss,
            "TverskyLoss": MyTverskyLoss,
            "FocalTverskyLoss": MyFocalTverskyLoss
        }

    def get_loss(self, loss_name, **loss_parameter):
        # Mixed loss인 경우
        if loss_name == "Mixed":
            losses = []
            weights = []
            
            # cfg.losses에서 설정 가져오기
            for loss_config in loss_parameter.get('losses', []):
                loss_fn = self.loss_classes.get(loss_config['name'])(**loss_config.get('params', {}))
                if loss_fn is not None:
                    losses.append(loss_fn)
                    weights.append(loss_config.get('weight', 1.0))
            
            return MixLoss(losses, weights)
        return self.loss_classes.get(loss_name, None)(**loss_parameter)