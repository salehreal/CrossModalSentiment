import os
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
from optimization.feature_selector import select_features


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(DEVICE)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, local_files_only=False)

FEATURE_SELECTION_METHOD = "HYBRID"


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
    sample_weights = [class_weights[lbl].item() for _, _, _, _, lbl, _ in ds]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return counts, class_weights, sampler


print("=== Loading Datasets ===")
train_dataset = MOSEICSDDataset("data/CMU-MOSEI", "train", tokenizer, max_len=64)
val_dataset   = MOSEICSDDataset("data/CMU-MOSEI", "val", tokenizer, max_len=64)

train_counts, class_weights_tensor, train_sampler = build_weighted_sampler(train_dataset, NUM_CLASSES)
val_counts, _ = compute_class_weights_from_dataset(val_dataset, NUM_CLASSES)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, collate_fn=multimodal_collate)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, collate_fn=multimodal_collate)


print("=== Initializing Models ===")

text_model   = TextFeatureExtractor(BERT_MODEL, TEXT_EMBED_DIM).to(device)
img_model    = ImageFeatureExtractor(35, IMG_EMBED_DIM, 256).to(device)
audio_model  = AudioFeatureExtractor(74, AUDIO_EMBED_DIM, 128).to(device)
emoji_extractor = EmojiFeatureExtractor(embed_dim=64, proj_dim=TEXT_EMBED_DIM).to(device)

fusion_model = AttentionFusion(
    TEXT_EMBED_DIM, IMG_EMBED_DIM, TEXT_EMBED_DIM, AUDIO_EMBED_DIM,
    FUSION_DIM, text_bias=0.10, emoji_bias=0.05, audio_bias=0.10
).to(device)

classifier = SentimentClassifier(FUSION_DIM, NUM_CLASSES).to(device)

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

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)


print("\n=== Extracting fused features for Feature Selection ===")

all_fused = []
all_labels = []

text_model.eval()
img_model.eval()
audio_model.eval()
emoji_extractor.eval()
fusion_model.eval()

with torch.no_grad():
    for batch in train_loader:
        input_ids, attention_mask, imgs, audios, labels_batch, raw_texts = batch

        raw_texts = [ensure_clean_text(t) for t in raw_texts]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        imgs = imgs.to(device)
        audios = audios.to(device)

        t = safe_normalize(text_model(input_ids, attention_mask))
        i = safe_normalize(img_model(imgs))
        a = safe_normalize(audio_model(audios))
        e = safe_normalize(emoji_extractor(raw_texts).to(device))

        fused = fusion_model(t, i, e, a)
        fused = torch.nan_to_num(fused)

        all_fused.append(fused.cpu())
        all_labels.append(labels_batch)

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, imgs, audios, labels_batch, raw_texts = batch

        raw_texts = [ensure_clean_text(t) for t in raw_texts]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        imgs = imgs.to(device)
        audios = audios.to(device)

        t = safe_normalize(text_model(input_ids, attention_mask))
        i = safe_normalize(img_model(imgs))
        a = safe_normalize(audio_model(audios))
        e = safe_normalize(emoji_extractor(raw_texts).to(device))

        fused = fusion_model(t, i, e, a)
        fused = torch.nan_to_num(fused)

        all_fused.append(fused.cpu())
        all_labels.append(labels_batch)

all_fused = torch.cat(all_fused, dim=0)
all_labels = torch.cat(all_labels, dim=0)

print("Fused feature matrix shape:", all_fused.shape)


print("\n=== Running Feature Selection (HYBRID) ===")

mask = select_features(
    all_fused.to(device),
    all_labels.to(device),
    method=FEATURE_SELECTION_METHOD,
    input_dim=FUSION_DIM,
    device=device,
    min_features=64
)

mask = mask.float().to(device)
mask = mask.unsqueeze(0)

print("Selected features:", int(mask.sum().item()))


def run_epoch(loader, train_mode=True):
    if train_mode:
        text_model.train(); img_model.train(); audio_model.train()
        emoji_extractor.train(); fusion_model.train(); classifier.train()
    else:
        text_model.eval(); img_model.eval(); audio_model.eval()
        emoji_extractor.eval(); fusion_model.eval(); classifier.eval()

    total_loss = 0.0
    y_true, y_pred = [], []

    for batch in loader:
        input_ids, attention_mask, imgs, audios, labels_batch, raw_texts = batch

        raw_texts = [ensure_clean_text(t) for t in raw_texts]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        imgs = imgs.to(device)
        audios = audios.to(device)
        labels_batch = labels_batch.to(device)

        with torch.set_grad_enabled(train_mode):
            t = safe_normalize(text_model(input_ids, attention_mask))
            i = safe_normalize(img_model(imgs))
            a = safe_normalize(audio_model(audios))
            e = safe_normalize(emoji_extractor(raw_texts).to(device))

            fused = fusion_model(t, i, e, a)
            fused = fused * mask

            logits = classifier(fused)
            loss = criterion(logits, labels_batch)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item() * labels_batch.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels_batch.detach().cpu().tolist())

    return total_loss / len(loader.dataset), evaluate(y_true, y_pred)


EPOCHS = 10
best_val_f1 = -1

print("\n=== Starting Training ===")
for epoch in range(EPOCHS):
    train_loss, train_metrics = run_epoch(train_loader, True)
    val_loss, val_metrics = run_epoch(val_loader, False)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.3f} | F1: {train_metrics['f1']:.3f} || "
        f"Val Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.3f} | F1: {val_metrics['f1']:.3f}"
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
            "mask": mask,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
        }, "model.pth")
        print(f"[INFO] Best model saved with F1={best_val_f1:.4f}")

print("Training finished.")
