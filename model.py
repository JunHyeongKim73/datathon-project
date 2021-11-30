import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EffNetModel(nn.Module):
    def __init__(self, model_name):
        super(EffNetModel, self).__init__()

        self.backbone = EfficientNet.from_pretrained(model_name, num_classes=150)
        
    def forward(self, x):
        x = self.backbone(x)
        
        return x