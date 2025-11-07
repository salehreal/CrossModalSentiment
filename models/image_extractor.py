import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class ImageFeatureExtractor(nn.Module):
    def __init__(self, output_dim=4096):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        self.features = vgg.features
        self.avgpool = vgg.avgpool

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
