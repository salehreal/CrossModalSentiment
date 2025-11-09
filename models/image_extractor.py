import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class ImageFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512, proj_dim=512):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.avgpool = vgg.avgpool

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, proj_dim),
            nn.ReLU()
        )

        self.proj = nn.Linear(proj_dim, output_dim)

        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.proj(x)
        x = F.normalize(x, p=2, dim=-1)
        return x
