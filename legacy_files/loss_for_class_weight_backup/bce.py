import torch
import torch.nn as nn

class MyBCELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MyBCELoss, self).__init__()

        ### 251227 추가: .yaml로부터 class weight를 받을 수 있도록 만듦.### 
        # pos_weight가 리스트로 들어오면 tensor로 변환
        if 'pos_weight' in kwargs and isinstance(kwargs['pos_weight'], list):
            kwargs['pos_weight'] = torch.tensor(kwargs['pos_weight'], dtype=torch.float32)
        #################################################################

        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, predictions, targets):
        return self.loss(predictions, targets)