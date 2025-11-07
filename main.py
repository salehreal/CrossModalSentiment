import torch
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
import os
from config import *
from models.text_extractor import TextFeatureExtractor
from models.image_extractor import ImageFeatureExtractor
from fusion.attention_fusion import AttentionFusion
from models.classifier import SentimentClassifier
from utils.emoji_processor import extract_emojis, emoji_sentiment_score

device = torch.device(DEVICE)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

text_model = TextFeatureExtractor().to(device)
img_model = ImageFeatureExtractor().to(device)
fusion_model = AttentionFusion(TEXT_EMBED_DIM, IMG_EMBED_DIM, FUSION_DIM).to(device)
classifier = SentimentClassifier(FUSION_DIM, NUM_CLASSES).to(device)
classifier.load_state_dict(torch.load("model.pth", map_location=device))

text_model.eval()
img_model.eval()
fusion_model.eval()
classifier.eval()

sample_text = input("Please enter your text: ")
image_choice = input("Please choose a picture (sad/happy/happy2/sad3...): ").strip()
sample_image_path = f"data/images/{image_choice}.jpg"

tokens = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
input_ids = tokens["input_ids"].to(device)
attention_mask = tokens["attention_mask"].to(device)
text_feat = text_model(input_ids, attention_mask)

if os.path.exists(sample_image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(sample_image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    img_feat = img_model(image_tensor)

    with torch.no_grad():
        fused_feat, alpha = fusion_model(text_feat, img_feat, return_alpha=True)
        output = classifier(fused_feat)
        text_only = classifier(fusion_model.text_proj(text_feat))
        img_only = classifier(fusion_model.img_proj(img_feat))
else:
    with torch.no_grad():
        fused_feat = text_feat
        output = classifier(fused_feat)
        alpha = torch.tensor([[1.0]])
        text_only = output
        img_only = torch.tensor([[0.0, 0.0]])

probs = torch.softmax(output, dim=1)
if probs[0][1] - probs[0][0] > 0.15:
    pred_label = 1
elif probs[0][0] - probs[0][1] > 0.15:
    pred_label = 0
else:
    pred_label = 0


label_map = {0: "Negative", 1: "Positive"}
sentiment = label_map[pred_label]

text_pred = torch.argmax(text_only, dim=1).item()
img_pred = torch.argmax(img_only, dim=1).item()

emojis = extract_emojis(sample_text)
emoji_score = emoji_sentiment_score(emojis)

print(f"Predicted Sentiment (fusion): {sentiment}")
