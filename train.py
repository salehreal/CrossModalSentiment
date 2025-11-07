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

device = torch.device(DEVICE)

text_model = TextFeatureExtractor().to(device)
img_model = ImageFeatureExtractor().to(device)
fusion_model = AttentionFusion(TEXT_EMBED_DIM, IMG_EMBED_DIM, FUSION_DIM).to(device)
classifier = SentimentClassifier(FUSION_DIM, NUM_CLASSES).to(device)

text_model.train()
img_model.train()
fusion_model.train()
classifier.train()

texts, images, labels = load_dataset("data/train.csv", "data/images")
train_texts, val_texts, train_images, val_images, train_labels, val_labels = train_test_split(
    texts, images, labels, test_size=0.2, random_state=42
)

counter = Counter(train_labels)
total = sum(counter.values())
class_weights = [total / counter[i] for i in sorted(counter)]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    list(text_model.parameters()) +
    list(img_model.parameters()) +
    list(fusion_model.parameters()) +
    list(classifier.parameters()),
    lr=1e-4
)

EPOCHS = 5
for epoch in range(EPOCHS):
    total_loss = 0.0
    y_true, y_pred = [], []

    for tokens, img, label in zip(train_texts, train_images, train_labels):
        input_ids, attention_mask = tokens
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        img = img.to(device)

        text_feat = text_model(input_ids, attention_mask)
        img_feat = img_model(img.unsqueeze(0))
        fused = fusion_model(text_feat, img_feat)

        if fused.dim() == 1:
            fused = fused.unsqueeze(0)

        output = classifier(fused)
        label_tensor = torch.tensor([label], dtype=torch.long).to(device)
        loss = criterion(output, label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        y_true.append(label)
        y_pred.append(torch.argmax(output, dim=1).item())

    train_metrics = evaluate(y_true, y_pred)
    print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Accuracy: {train_metrics['accuracy']:.2f} | F1: {train_metrics['f1']:.2f}")

    classifier.eval()
    with torch.no_grad():
        val_true, val_pred = [], []
        for tokens, img, label in zip(val_texts, val_images, val_labels):
            input_ids, attention_mask = tokens
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            img = img.to(device)

            text_feat = text_model(input_ids, attention_mask)
            img_feat = img_model(img.unsqueeze(0))
            fused = fusion_model(text_feat, img_feat)

            if fused.dim() == 1:
                fused = fused.unsqueeze(0)

            output = classifier(fused)
            val_true.append(label)
            val_pred.append(torch.argmax(output, dim=1).item())

        val_metrics = evaluate(val_true, val_pred)
        print(f"Validation | Accuracy: {val_metrics['accuracy']:.2f} | F1: {val_metrics['f1']:.2f}")
    classifier.train()

torch.save(classifier.state_dict(), "model.pth")
print("Model saved as model.pth")
