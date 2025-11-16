import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from collate_fn import multimodal_collate
from utils.metrics import evaluate
from optimization.feature_selector import FeatureSelectorGA

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(DEVICE)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, local_files_only=False)

def ensure_clean_text(x):
    s = x if isinstance(x, str) else str(x)
    s = s.strip()
    return s if s else "neutral"

def compute_class_weights_from_dataset(ds, num_classes):
    from collections import Counter
    counter = Counter()
    for _, _, _, _, lbl, _ in ds:
        counter[lbl] += 1

    counts = [counter.get(i, 0) for i in range(num_classes)]
    counts = [c if c > 0 else 1 for c in counts]
    inv_freq = [1.0 / c for c in counts]
    norm = sum(inv_freq)
    weights = [w * num_classes / norm for w in inv_freq]
    return counts, torch.tensor(weights, dtype=torch.float32)

def build_weighted_sampler(ds, num_classes):
    counts, class_weights = compute_class_weights_from_dataset(ds, num_classes)
    sample_weights = []
    for _, _, _, _, lbl, _ in ds:
        sample_weights.append(class_weights[lbl].item())
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return counts, class_weights, sampler

print("=== Loading Datasets ===")
train_dataset = MOSEICSDDataset(sdk_path="data/CMU-MOSEI", split="train", tokenizer=tokenizer, max_len=64)
val_dataset   = MOSEICSDDataset(sdk_path="data/CMU-MOSEI", split="val",   tokenizer=tokenizer, max_len=64)

train_counts, class_weights_tensor, train_sampler = build_weighted_sampler(train_dataset, NUM_CLASSES)
val_counts, _ = compute_class_weights_from_dataset(val_dataset, NUM_CLASSES)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print(f"Train class counts: {train_counts}")
print(f"Val class counts:   {val_counts}")
print(f"Class weights (criterion): {class_weights_tensor.tolist()}")

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=train_sampler,
    collate_fn=multimodal_collate,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=multimodal_collate,
    num_workers=0
)

print("=== Initializing Models ===")
text_model   = TextFeatureExtractor(bert_model=BERT_MODEL, output_dim=TEXT_EMBED_DIM).to(device)
img_model    = ImageFeatureExtractor(input_dim=35, output_dim=IMG_EMBED_DIM, proj_dim=256).to(device)
audio_model  = AudioFeatureExtractor(input_dim=74, output_dim=AUDIO_EMBED_DIM, proj_dim=128).to(device)
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
classifier   = SentimentClassifier(FUSION_DIM, NUM_CLASSES).to(device)

feature_selector = FeatureSelectorGA(input_dim=FUSION_DIM, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

optimizer = optim.Adam(
    list(text_model.parameters()) +
    list(img_model.parameters()) +
    list(audio_model.parameters()) +
    list(emoji_extractor.parameters()) +
    list(fusion_model.parameters()) +
    list(classifier.parameters()),
    lr=2e-5
)

try:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)
except TypeError:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

def run_epoch(loader, train_mode=True):
    if train_mode:
        text_model.train(); img_model.train(); audio_model.train()
        emoji_extractor.train(); fusion_model.train(); classifier.train()
    else:
        text_model.eval(); img_model.eval(); audio_model.eval()
        emoji_extractor.eval(); fusion_model.eval(); classifier.eval()

    total_loss = 0.0
    y_true, y_pred = [], []

    for batch_idx, batch in enumerate(loader):
        input_ids, attention_mask, imgs, audios, labels_batch, raw_texts = batch

        raw_texts = [ensure_clean_text(t) for t in raw_texts]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        imgs = imgs.to(device)
        audios = audios.to(device)
        labels_batch = labels_batch.to(device)

        with torch.set_grad_enabled(train_mode):
            text_feat  = text_model(input_ids, attention_mask)
            img_feat   = img_model(imgs)
            audio_feat = audio_model(audios)
            emoji_feat = emoji_extractor(raw_texts).to(device)

            text_feat  = torch.nan_to_num(text_feat,  nan=0.0, posinf=0.0, neginf=0.0)
            img_feat   = torch.nan_to_num(img_feat,   nan=0.0, posinf=0.0, neginf=0.0)
            audio_feat = torch.nan_to_num(audio_feat, nan=0.0, posinf=0.0, neginf=0.0)
            emoji_feat = torch.nan_to_num(emoji_feat, nan=0.0, posinf=0.0, neginf=0.0)

            text_feat  = safe_normalize(text_feat)
            img_feat   = safe_normalize(img_feat)
            audio_feat = safe_normalize(audio_feat)
            emoji_feat = safe_normalize(emoji_feat)

            fused = fusion_model(text_feat, img_feat, emoji_feat, audio_feat)
            fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

            fused = feature_selector.apply(fused)

            logits = classifier(fused)
            loss = criterion(logits, labels_batch)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(text_model.parameters()) +
                    list(img_model.parameters()) +
                    list(audio_model.parameters()) +
                    list(emoji_extractor.parameters()) +
                    list(fusion_model.parameters()) +
                    list(classifier.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

        total_loss += loss.item() * labels_batch.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels_batch.detach().cpu().tolist())

        if train_mode and batch_idx % 200 == 0 and batch_idx > 0:
            print(f"[Train] batch {batch_idx} | loss {loss.item():.6f}")

    metrics = evaluate(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics

EPOCHS = 10
best_val_f1 = -1.0

print("=== Starting Training ===")
for epoch in range(EPOCHS):
    train_loss, train_metrics = run_epoch(train_loader, train_mode=True)
    val_loss,   val_metrics   = run_epoch(val_loader,   train_mode=False)

    try:
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after != lr_before:
            print(f"[Scheduler] LR reduced: {lr_before:.6e} -> {lr_after:.6e}")
    except Exception as e:
        print(f"[Scheduler] step error: {e}", file=sys.stderr)

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.3f} | F1: {train_metrics['f1']:.3f} || "
        f"Val  Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.3f} | F1: {val_metrics['f1']:.3f}"
    )

    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        torch.save({
            "epoch": epoch,
            "text_model": text_model.state_dict(),
            "img_model": img_model.state_dict(),
            "audio_model": audio_model.state_dict(),
            "emoji_extractor": emoji_extractor.state_dict(),
            "fusion_model": fusion_model.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_metrics": val_metrics,
        }, "model.pth")
        print(f"[INFO] Best model saved with val F1={best_val_f1:.4f}")

print("Training finished.")
