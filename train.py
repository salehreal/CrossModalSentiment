import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from models.text_extractor import TextFeatureExtractor
from models.image_extractor import ImageFeatureExtractor
from fusion.attention_fusion import AttentionFusion
from models.classifier import SentimentClassifier
from utils.dataset_loader import load_dataset
from utils.metrics import evaluate
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

device = torch.device(DEVICE)

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    images = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    images = torch.stack(images)

    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids_padded, attention_masks_padded, images, labels

class CommentsDataset(Dataset):
    def __init__(self, texts, images, labels):
        self.texts = texts
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids, attention_mask = self.texts[idx]
        input_ids = input_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        image = self.images[idx]
        label = int(self.labels[idx])
        return input_ids, attention_mask, image, label

text_model = TextFeatureExtractor().to(device)
img_model = ImageFeatureExtractor(output_dim=IMG_EMBED_DIM).to(device)
fusion_model = AttentionFusion(TEXT_EMBED_DIM, IMG_EMBED_DIM, FUSION_DIM).to(device)
classifier = SentimentClassifier(FUSION_DIM, NUM_CLASSES).to(device)

text_model.train()
img_model.train()
fusion_model.train()
classifier.train()

texts, images, labels = load_dataset("user_comments.csv", "images/")
train_texts, val_texts, train_images, val_images, train_labels, val_labels = train_test_split(
    texts, images, labels, test_size=0.2, random_state=42, stratify=labels
)

train_ds = CommentsDataset(train_texts, train_images, train_labels)
val_ds = CommentsDataset(val_texts, val_images, val_labels)

BATCH_SIZE = 16
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

counter = Counter(train_labels)
weights = [counter[i] for i in sorted(counter)]
class_weights = torch.tensor([1.0 / w for w in weights], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    list(text_model.parameters()) +
    list(img_model.parameters()) +
    list(fusion_model.parameters()) +
    list(classifier.parameters()),
    lr=1e-5
)

EPOCHS = 5
for epoch in range(EPOCHS):
    total_loss = 0.0
    y_true, y_pred = [], []
    text_model.train()
    img_model.train()
    fusion_model.train()
    classifier.train()

    for batch in train_loader:
        input_ids, attention_mask, imgs, labels_batch = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        imgs = imgs.to(device)
        labels_batch = labels_batch.to(device, dtype=torch.long)

        text_feat = text_model(input_ids, attention_mask)
        img_feat = img_model(imgs)

        text_feat = F.normalize(text_feat, p=2, dim=-1)
        img_feat = F.normalize(img_feat, p=2, dim=-1)

        if np.random.rand() < 0.03:
            print(f"[DEBUG] text norm mean: {text_feat.norm(dim=-1).mean().item():.4f}, img norm mean: {img_feat.norm(dim=-1).mean().item():.4f}")

        fused = fusion_model(text_feat, img_feat)
        output = classifier(fused)

        loss = criterion(output, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels_batch.size(0)
        preds = torch.argmax(output, dim=1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels_batch.detach().cpu().tolist())

    train_metrics = evaluate(y_true, y_pred)
    avg_loss = total_loss / len(train_ds)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Accuracy: {train_metrics['accuracy']:.2f} | F1: {train_metrics['f1']:.2f}")

    text_model.eval()
    img_model.eval()
    fusion_model.eval()
    classifier.eval()

    with torch.no_grad():
        val_true, val_pred = [], []
        for batch in val_loader:
            input_ids, attention_mask, imgs, labels_batch = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            imgs = imgs.to(device)
            labels_batch = labels_batch.to(device, dtype=torch.long)

            text_feat = text_model(input_ids, attention_mask)
            img_feat = img_model(imgs)

            text_feat = F.normalize(text_feat, p=2, dim=-1)
            img_feat = F.normalize(img_feat, p=2, dim=-1)

            fused = fusion_model(text_feat, img_feat)
            output = classifier(fused)

            preds = torch.argmax(output, dim=1).detach().cpu().tolist()
            val_pred.extend(preds)
            val_true.extend(labels_batch.detach().cpu().tolist())

        val_metrics = evaluate(val_true, val_pred)
        print(f"Validation | Accuracy: {val_metrics['accuracy']:.2f} | F1: {val_metrics['f1']:.2f}")
        print(f"Validation Predictions → Positive: {val_pred.count(1)}, Negative: {val_pred.count(0)}")

torch.save({
    "text_model": text_model.state_dict(),
    "img_model": img_model.state_dict(),
    "fusion_model": fusion_model.state_dict(),
    "classifier": classifier.state_dict()
}, "model.pth")

print("All models saved in model.pth")
