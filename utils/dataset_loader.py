import torch
from torch.utils.data import Dataset
import numpy as np
from mmsdk import mmdatasdk

def _to_str_token(x):
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return x.decode("latin1", errors="ignore")
    return str(x)

class MOSEICSDDataset(Dataset):
    def __init__(self, sdk_path, split="train", tokenizer=None, max_len=64):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.sdk = mmdatasdk.mmdataset({
            "text": f"{sdk_path}/languages/CMU_MOSEI_TimestampedWords.csd",
            "vision": f"{sdk_path}/visuals/CMU_MOSEI_VisualFacet42.csd",
            "audio": f"{sdk_path}/acoustics/CMU_MOSEI_COVAREP.csd",
            "label": f"{sdk_path}/labels/CMU_MOSEI_Labels.csd",
        })

        all_ids = list(self.sdk["label"].data.keys())
        filtered = [vid for vid in all_ids
                    if vid in self.sdk["text"].data
                    and vid in self.sdk["vision"].data
                    and vid in self.sdk["audio"].data]

        n = len(filtered)
        if split == "train":
            self.ids = filtered[: int(0.8 * n)]
        elif split == "val":
            self.ids = filtered[int(0.8 * n): int(0.9 * n)]
        elif split == "test":
            self.ids = filtered[int(0.9 * n):]
        else:
            raise ValueError("Invalid split")
        print(f"[INFO] Loaded {len(self.ids)} samples for {split}")

    def __len__(self):
        return len(self.ids)

    def safe_mean(self, arr, fallback_dim):
        if arr is None:
            return np.zeros(fallback_dim, dtype=np.float32)
        arr = np.array(arr, dtype=np.float32)
        if arr.size == 0:
            return np.zeros(fallback_dim, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.mean(arr, axis=0).astype(np.float32)

    def extract_text(self, vid):
        words = self.sdk["text"].data[vid]["features"]
        tokens = []
        for w in words:
            token = w[0] if isinstance(w, (list, tuple, np.ndarray)) else w
            token = _to_str_token(token).strip()
            if token and token.lower() != "sp":
                tokens.append(token)
        text = " ".join(tokens).strip()
        return text if text else "neutral"

    def __getitem__(self, idx):
        vid = self.ids[idx]
        raw_text = self.extract_text(vid)
        raw_text = _to_str_token(raw_text).strip()

        try:
            label_val = float(self.sdk["label"].data[vid]["features"][0][0])
            label = 1 if label_val > 0 else 0
        except Exception:
            label = 0

        try:
            vision = self.sdk["vision"].data[vid]["features"]
            vision_avg = self.safe_mean(vision, 35)
        except Exception:
            vision_avg = np.zeros(35, dtype=np.float32)
        vision_tensor = torch.tensor(vision_avg, dtype=torch.float32)

        try:
            audio = self.sdk["audio"].data[vid]["features"]
            audio_avg = self.safe_mean(audio, 74)
        except Exception:
            audio_avg = np.zeros(74, dtype=np.float32)
        audio_tensor = torch.tensor(audio_avg, dtype=torch.float32)

        if self.tokenizer:
            enc = self.tokenizer(
                raw_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_len, dtype=torch.long)

        return input_ids, attention_mask, vision_tensor, audio_tensor, label, raw_text
