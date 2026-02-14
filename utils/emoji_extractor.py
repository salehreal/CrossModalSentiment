import torch
import torch.nn as nn
import re

EMOJI_PATTERN = re.compile(
    r"[\U0001F300-\U0001F6FF\U0001F900-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF]"
)

class EmojiFeatureExtractor(nn.Module):
    def __init__(self, embed_dim=64, proj_dim=256, vocab=None):
        super().__init__()

        default_vocab = [
            "😀","😂","🤣","😊","😍","💕","👍","🔥","✨",
            "😢","😭","💔","👎","😡","🤬"
        ]

        self.vocab = vocab if vocab is not None else default_vocab
        self.emoji2idx = {e: i for i, e in enumerate(self.vocab)}

        self.emb = nn.Embedding(len(self.vocab) + 1, embed_dim)

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU()
        )

    def forward(self, raw_texts):
        device = next(self.emb.parameters()).device
        batch_indices = []

        for t in raw_texts:
            if not isinstance(t, str):
                t = str(t)

            emojis = EMOJI_PATTERN.findall(t)

            if not emojis:
                unk_idx = len(self.vocab)
                batch_indices.append(
                    torch.tensor([unk_idx], dtype=torch.long, device=device)
                )
                continue

            idxs = [
                self.emoji2idx.get(e, len(self.vocab))
                for e in emojis
            ]
            batch_indices.append(
                torch.tensor(idxs, dtype=torch.long, device=device)
            )

        embeds = []
        for idxs in batch_indices:
            emb = self.emb(idxs.to(device))
            if emb.numel() == 0:
                emb_mean = torch.zeros(self.emb.embedding_dim, device=device)
            else:
                emb_mean = emb.mean(dim=0)
            embeds.append(emb_mean)

        E = torch.stack(embeds, dim=0)
        E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
        return self.proj(E)
