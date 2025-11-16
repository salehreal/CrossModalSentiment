import torch
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
import random

def ensure_clean_text(x):

    import numpy as np
    def to_str_token(tok):
        if isinstance(tok, bytes):
            try:
                return tok.decode("utf-8", errors="ignore")
            except Exception:
                return tok.decode("latin1", errors="ignore")
        return str(tok)

    if isinstance(x, (list, tuple, np.ndarray)):
        flat = []
        for t in x:
            if isinstance(t, (list, tuple, np.ndarray)):
                for s in t:
                    flat.append(to_str_token(s))
            else:
                flat.append(to_str_token(t))
        s = " ".join([w for w in (w.strip() for w in flat) if w and w.lower() != "sp"])
        return s if s.strip() else "neutral"
    s = to_str_token(x).strip()
    return s if s else "neutral"

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

text_model = TextFeatureExtractor(bert_model=BERT_MODEL, output_dim=TEXT_EMBED_DIM).to(device)
img_model = ImageFeatureExtractor(input_dim=35, output_dim=IMG_EMBED_DIM, proj_dim=256).to(device)
audio_model = AudioFeatureExtractor(input_dim=74, output_dim=AUDIO_EMBED_DIM, proj_dim=128).to(device)
emoji_extractor = EmojiFeatureExtractor(embed_dim=64, proj_dim=TEXT_EMBED_DIM).to(device)
fusion_model = AttentionFusion(
    text_dim=TEXT_EMBED_DIM,
    img_dim=IMG_EMBED_DIM,
    emoji_dim=TEXT_EMBED_DIM,
    audio_dim=AUDIO_EMBED_DIM,
    fusion_dim=FUSION_DIM,
    text_bias=0.10,
    emoji_bias=0.05,
    audio_bias=0.10
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
        print("Models loaded successfully from checkpoint.")
    except Exception as e:
        print(f"Warning while loading model weights: {e}")

text_model.eval(); img_model.eval(); audio_model.eval()
emoji_extractor.eval(); fusion_model.eval(); classifier.eval()

test_ds = MOSEICSDDataset(sdk_path="data/CMU-MOSEI", split="test", tokenizer=tokenizer, max_len=64)
label_map = {0: "Negative", 1: "Positive"}

indices = random.sample(range(len(test_ds)), 5)
print(f"\n=== Testing on {indices} samples ===\n")

for i in indices:
    input_ids, attention_mask, vision, audio, true_label, raw_text = test_ds[i]

    raw_text = ensure_clean_text(raw_text)

    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    vision = vision.unsqueeze(0).to(device)
    audio = audio.unsqueeze(0).to(device)

    with torch.no_grad():
        text_feat = text_model(input_ids, attention_mask)
        img_feat = img_model(vision)
        audio_feat = audio_model(audio)
        emoji_feat = emoji_extractor([raw_text if raw_text.strip() else "neutral"])

        text_feat = safe_normalize(torch.nan_to_num(text_feat))
        img_feat = safe_normalize(torch.nan_to_num(img_feat))
        audio_feat = safe_normalize(torch.nan_to_num(audio_feat))
        emoji_feat = safe_normalize(torch.nan_to_num(emoji_feat))

        fused, alpha = fusion_model(text_feat, img_feat, emoji_feat, audio_feat, return_alpha=True)
        output = classifier(fused)
        probs = torch.softmax(output, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred_label].item()

    print(f"----- Sample {i+1} -----")
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
