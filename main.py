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
import torch.nn.functional as F

device = torch.device(DEVICE)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

checkpoint = torch.load("model.pth", map_location=device)

text_model = TextFeatureExtractor().to(device)
text_model.load_state_dict(checkpoint["text_model"])
text_model.eval()

img_model = ImageFeatureExtractor(output_dim=IMG_EMBED_DIM).to(device)
img_model.load_state_dict(checkpoint["img_model"])
img_model.eval()

fusion_model = AttentionFusion(TEXT_EMBED_DIM, IMG_EMBED_DIM, FUSION_DIM, alpha_range=(0.7, 1.0)).to(device)
fusion_model.load_state_dict(checkpoint["fusion_model"])
fusion_model.eval()

classifier = SentimentClassifier(FUSION_DIM, NUM_CLASSES).to(device)
classifier.load_state_dict(checkpoint["classifier"])
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_choice = input("Choose an image to comment on (img1, img2, img3, img4, img5): ").strip()
sample_text = input("Enter your comment: ")
emotion_image_choice = input("Choose an emotional image (happy, happy2, ..., sad5) or leave blank: ").strip()

main_image_path = f"images/{image_choice}.jpg"
emotional_image_path = f"data/images/{emotion_image_choice}.jpg" if emotion_image_choice else None

tokens = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
input_ids = tokens["input_ids"].to(device)
attention_mask = tokens["attention_mask"].to(device)
with torch.no_grad():
    text_feat = text_model(input_ids, attention_mask)
    text_feat = F.normalize(text_feat, p=2, dim=-1)

main_feat, emo_feat = None, None

if os.path.exists(main_image_path):
    image = Image.open(main_image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        main_feat = img_model(image_tensor)
        main_feat = F.normalize(main_feat, p=2, dim=-1)
    # print(f"Main image feature norm: {main_feat.norm().item():.2f}")

if emotional_image_path and os.path.exists(emotional_image_path):
    emo_image = Image.open(emotional_image_path).convert("RGB")
    emo_tensor = transform(emo_image).unsqueeze(0).to(device)
    with torch.no_grad():
        emo_feat = img_model(emo_tensor)
        emo_feat = F.normalize(emo_feat, p=2, dim=-1)
    # print(f"Emotional image feature norm: {emo_feat.norm().item():.2f}")

if main_feat is not None and emo_feat is not None:
    img_feat = 0.5 * main_feat + 0.5 * emo_feat
elif main_feat is not None:
    img_feat = main_feat
elif emo_feat is not None:
    img_feat = emo_feat
else:
    img_feat = None

with torch.no_grad():
    if img_feat is not None:
        fused_feat, alpha = fusion_model(text_feat, img_feat, return_alpha=True)
        output = classifier(fused_feat)
    else:
        try:
            fused_text = fusion_model.text_proj(text_feat)
            fused_feat = fused_text
        except Exception:
            fused_feat = text_feat
        output = classifier(fused_feat)
        alpha = torch.tensor([[1.0]])

probs = torch.softmax(output, dim=1)
pred_label = torch.argmax(probs, dim=1).item()
label_map = {0: "Negative", 1: "Positive"}
sentiment = label_map[pred_label]

emojis = extract_emojis(sample_text)
emoji_score = emoji_sentiment_score(emojis)

print(f"\nPredicted Sentiment: {sentiment}")
print(f"Emoji Sentiment Score: {emoji_score:.2f}")
print(f"Attention Weight (text): {alpha.item():.2f}")

print("\nComparative Test of Inputs:")

with torch.no_grad():
    try:
        text_only_feat = fusion_model.text_proj(text_feat) if hasattr(fusion_model, "text_proj") else text_feat
        text_only_out = classifier(text_only_feat)
        text_only_pred = torch.argmax(torch.softmax(text_only_out, dim=1), dim=1).item()
        # print(f"Text Only → {label_map[text_only_pred]}")
    except Exception:
        pass
        # print("Text Only → (couldn't compute)")

    if img_feat is not None:
        try:
            img_only_feat = fusion_model.img_proj(img_feat) if hasattr(fusion_model, "img_proj") else img_feat
            img_only_out = classifier(img_only_feat)
            img_only_pred = torch.argmax(torch.softmax(img_only_out, dim=1), dim=1).item()
            # print(f"Image Only → {label_map[img_only_pred]}")
        except Exception:
            pass
            # print("Image Only → (couldn't compute)")
    else:
        pass
        # print("Image Only → (no image)")

    if main_feat is not None:
        fused_main, alpha_main = fusion_model(text_feat, main_feat, return_alpha=True)
        main_out = classifier(fused_main)
        main_pred = torch.argmax(torch.softmax(main_out, dim=1), dim=1).item()
        # print(f"Text + Main Image → {label_map[main_pred]}")
        # print(f"Attention (text) for Main Image: {alpha_main.item():.2f}")
    else:
        pass
        # print("Main image not found.")

    if main_feat is not None and emo_feat is not None:
        combined_feat = 0.5 * main_feat + 0.5 * emo_feat
        fused_full, alpha_full = fusion_model(text_feat, combined_feat, return_alpha=True)
        full_out = classifier(fused_full)
        full_pred = torch.argmax(torch.softmax(full_out, dim=1), dim=1).item()
        print(f"Text + Main + Emotional Image → {label_map[full_pred]}")
        # print(f"Attention (text) for Full Fusion: {alpha_full.item():.2f}")
    else:
        print("Emotional image not provided or missing.")
