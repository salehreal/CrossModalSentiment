import pandas as pd
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import torch
import os

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_dataset(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    texts, images, labels = [], [], []

    for _, row in df.iterrows():
        text = str(row["text"])
        label = int(row["label"])
        image_name = str(row["image"])
        image_path = os.path.join(image_dir, image_name)

        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image)
        except Exception as e:
            print(f"Failed to load image: {image_path} → {e}")
            image_tensor = torch.zeros((3, 224, 224))

        texts.append((input_ids, attention_mask))
        images.append(image_tensor)
        labels.append(label)

    return texts, images, labels
