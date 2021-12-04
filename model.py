import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EffNetModel(nn.Module):
    def __init__(self, model_name):
        super(EffNetModel, self).__init__()

        self.backbone = EfficientNet.from_pretrained(model_name, num_classes=150)
        
        # dropout과 relu는 nn.를 써도 되고 nn.functional를 써도 되지만
        # nn.를 쓸 때는 먼저 선언을 해야한다
        # nn. 내부에 nn.functional을 불러오는 식으로 되어있어서
        # 클래스가 중복돼도 상관없는 것이다
        
    def forward(self, x):
        x = self.backbone(x)
        
        return x