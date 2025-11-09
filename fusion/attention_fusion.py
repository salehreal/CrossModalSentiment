import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, text_dim, img_dim, fusion_dim, alpha_range=(0.5, 0.9)):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.img_proj = nn.Linear(img_dim, fusion_dim)

        self.attn_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(fusion_dim)
        )

        self.alpha_min, self.alpha_max = alpha_range

    def forward(self, text_feat, img_feat, return_alpha=False):
        text_proj = self.text_proj(text_feat)
        img_proj = self.img_proj(img_feat)

        concat = torch.cat([text_proj, img_proj], dim=1)
        alpha = self.attn_layer(concat)

        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha

        fused = alpha * text_proj + (1 - alpha) * img_proj

        fused = F.normalize(fused, p=2, dim=-1)
        fused_out = self.fusion_mlp(fused)

        return (fused_out, alpha) if return_alpha else fused_out
