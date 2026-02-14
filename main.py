import torch
import random
import numpy as np
from transformers import BertTokenizer

from config import *
from models.text_extractor import TextFeatureExtractor
from models.image_extractor import ImageFeatureExtractor
from models.audio_extractor import AudioFeatureExtractor
from fusion.attention_fusion import AttentionFusion
from models.classifier import SentimentClassifier
from utils.emoji_extractor import EmojiFeatureExtractor
from utils.dataset_loader import MOSEICSDDataset
from utils.tensor_utils import safe_normalize


def ensure_clean_text(x):
    if isinstance(x, str):
        s = x.strip()
        return s if s else "neutral"
    return "neutral"


device = torch.device(DEVICE)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, local_files_only=False)

print("Cross-Modal Sentiment Analysis - Testing")
print("=" * 50)

checkpoint_path = "model.pth"
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from {checkpoint_path}")
except Exception as e:
    print(f"Could not load checkpoint '{checkpoint_path}': {e}")
    checkpoint = None

text_model = TextFeatureExtractor(BERT_MODEL, TEXT_EMBED_DIM).to(device)
img_model = ImageFeatureExtractor(35, IMG_EMBED_DIM, 256).to(device)
audio_model = AudioFeatureExtractor(74, AUDIO_EMBED_DIM, 128).to(device)
emoji_extractor = EmojiFeatureExtractor(embed_dim=64, proj_dim=TEXT_EMBED_DIM).to(device)

fusion_model = AttentionFusion(
    TEXT_EMBED_DIM, IMG_EMBED_DIM, TEXT_EMBED_DIM, AUDIO_EMBED_DIM,
    FUSION_DIM, text_bias=0.10, emoji_bias=0.05, audio_bias=0.10
).to(device)

classifier = SentimentClassifier(FUSION_DIM, NUM_CLASSES).to(device)

if checkpoint is not None:
    try:
        if "text_model" in checkpoint: text_model.load_state_dict(checkpoint["text_model"])
        if "img_model" in checkpoint: img_model.load_state_dict(checkpoint["img_model"])
        if "audio_model" in checkpoint: audio_model.load_state_dict(checkpoint["audio_model"])
        if "emoji_extractor" in checkpoint: emoji_extractor.load_state_dict(checkpoint["emoji_extractor"])
        if "fusion_model" in checkpoint: fusion_model.load_state_dict(checkpoint["fusion_model"])
        if "classifier" in checkpoint: classifier.load_state_dict(checkpoint["classifier"])
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Warning while loading model weights: {e}")

if checkpoint is not None and "mask" in checkpoint:
    mask = checkpoint["mask"].to(device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    print("Feature selection mask loaded.")
else:
    mask = None
    print("No mask found in checkpoint. Running without feature selection.")

text_model.eval()
img_model.eval()
audio_model.eval()
emoji_extractor.eval()
fusion_model.eval()
classifier.eval()

test_ds = MOSEICSDDataset(
    sdk_path="data/CMU-MOSEI",
    split="test",
    tokenizer=tokenizer,
    max_len=64
)

label_map = {0: "Negative", 1: "Positive"}

indices = random.sample(range(len(test_ds)), 5)
print(f"\n=== Testing on samples: {indices} ===\n")

for i in indices:
    input_ids, attention_mask, vision, audio, true_label, raw_text = test_ds[i]

    raw_text = ensure_clean_text(raw_text)

    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    vision = vision.unsqueeze(0).to(device)
    audio = audio.unsqueeze(0).to(device)

    with torch.no_grad():
        t = safe_normalize(text_model(input_ids, attention_mask))
        i_feat = safe_normalize(img_model(vision))
        a_feat = safe_normalize(audio_model(audio))
        e_feat = safe_normalize(emoji_extractor([raw_text]).to(device))

        fused, alpha = fusion_model(
            t, i_feat, e_feat, a_feat,
            return_alpha=True
        )

        if mask is not None:
            fused = fused * mask

        output = classifier(fused)
        probs = torch.softmax(output, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred_label].item()

    print(f"----- Sample {i} -----")
    print(f"Text: {raw_text}")
    print(f"True Label: {label_map[true_label]}")
    print(f"Predicted: {label_map[pred_label]}")
    print(f"Confidence: {conf:.3f}")
    print("Attention Weights:")
    print(f"  Text:   {alpha[0,0].item():.3f}")
    print(f"  Image:  {alpha[0,1].item():.3f}")
    print(f"  Emoji:  {alpha[0,2].item():.3f}")
    print(f"  Audio:  {alpha[0,3].item():.3f}")
    print()

print("=== Testing completed! ===")
