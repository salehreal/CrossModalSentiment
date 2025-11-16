import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tensor_utils import safe_normalize

class AttentionFusion(nn.Module):
    def __init__(self, text_dim, img_dim, emoji_dim, audio_dim, fusion_dim, 
                 text_bias=0.0, emoji_bias=0.0, audio_bias=0.0):
        super().__init__()
        
        self.fusion_dim = fusion_dim

        self.value_text = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.value_img = nn.Sequential(
            nn.Linear(img_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.value_emoji = nn.Sequential(
            nn.Linear(emoji_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.value_audio = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.attention_net = nn.Sequential(
            nn.Linear(text_dim + img_dim + emoji_dim + audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

        self.register_buffer('modality_bias', torch.tensor([text_bias, 0.0, emoji_bias, audio_bias]))

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, text_feat, img_feat, emoji_feat, audio_feat, return_alpha=False):
        batch_size = text_feat.size(0)

        text_feat = safe_normalize(torch.nan_to_num(text_feat, nan=0.0, posinf=1.0, neginf=-1.0))
        img_feat = safe_normalize(torch.nan_to_num(img_feat, nan=0.0, posinf=1.0, neginf=-1.0))
        emoji_feat = safe_normalize(torch.nan_to_num(emoji_feat, nan=0.0, posinf=1.0, neginf=-1.0))
        audio_feat = safe_normalize(torch.nan_to_num(audio_feat, nan=0.0, posinf=1.0, neginf=-1.0))

        V_text = self.value_text(text_feat)
        V_img = self.value_img(img_feat)
        V_emoji = self.value_emoji(emoji_feat)
        V_audio = self.value_audio(audio_feat)

        combined_features = torch.cat([text_feat, img_feat, emoji_feat, audio_feat], dim=1)
        attention_logits = self.attention_net(combined_features)

        attention_logits = attention_logits + self.modality_bias.unsqueeze(0)

        alpha = F.softmax(attention_logits, dim=1)

        fused = (
            alpha[:, 0].unsqueeze(1) * V_text +
            alpha[:, 1].unsqueeze(1) * V_img +
            alpha[:, 2].unsqueeze(1) * V_emoji +
            alpha[:, 3].unsqueeze(1) * V_audio
        )

        fused = self.fusion_layer(fused)
        fused = safe_normalize(torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0))
        
        if return_alpha:
            return fused, alpha
        return fused