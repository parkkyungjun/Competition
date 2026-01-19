import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.models import efficientnet_b2, EfficientNet, EfficientNet_B2_Weights

__all__ = ['CustomEfficientNet', 'efficientnetb2_custom']


class CustomEfficientNet(nn.Module):
    def __init__(self, base_model: EfficientNet, num_classes: int = 1):
        super().__init__()
        
        self.stem = base_model.features[0]
        self.blocks = base_model.features[1:]
        
        self.avgpool = base_model.avgpool
        
        in_features = base_model.classifier[1].in_features
        dropout_p = base_model.classifier[0].p
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)

    def forward(self, x):
        NPR  = x - self.interpolate(x, 0.5)
        scaled_NPR = NPR * 2.0 / 3.0
        
        x = self.stem(scaled_NPR)
        x = self.blocks(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


def efficientnetb2_custom(pretrained: bool = False, num_classes=1):
    base_model = efficientnet_b2(weights=EfficientNet_B2_Weights)
    
    model = CustomEfficientNet(base_model, num_classes=num_classes)
    return model
