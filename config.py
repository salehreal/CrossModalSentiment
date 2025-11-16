import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_MODEL = "bert-base-uncased"
TEXT_EMBED_DIM = 128
IMG_EMBED_DIM = 128
AUDIO_EMBED_DIM = 128
FUSION_DIM = 256
NUM_CLASSES = 2
EPOCHS = 15
BATCH_SIZE = 8
GRADIENT_CLIP = 1.0
WEIGHT_DECAY = 1e-4
TEXT_LR = 2e-5
IMG_LR = 1e-4
AUDIO_LR = 1e-4
EMOJI_LR = 1e-4
FUSION_LR = 1e-4
CLASSIFIER_LR = 1e-4
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_DELTA = 0.002
MAX_SEQ_LEN = 64

if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.7)