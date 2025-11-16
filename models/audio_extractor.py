import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioFeatureExtractor(nn.Module):
    def __init__(self, input_dim=74, output_dim=256, proj_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, proj_dim),
            nn.ReLU()
        )

        self.proj = nn.Linear(proj_dim, output_dim)

        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.fc(x)
        x = self.proj(x)
        return x
