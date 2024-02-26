import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

IMAGENET_WEIGHTS = "IMAGENET1K_V1"


class DenseNet(nn.Module):
    def __init__(self, pretrained=True, num_outputs=1):
        super().__init__()

        base_model = models.densenet121(weights=IMAGENET_WEIGHTS if pretrained else None)
        num_features = 1024

        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class VGGNet(nn.Module):
    def __init__(self, pretrained=True, num_outputs=1):
        super().__init__()

        base_model = models.vgg19(weights=IMAGENET_WEIGHTS if pretrained else None)
        num_features = 512

        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
