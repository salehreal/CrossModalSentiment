import torch
from torch.nn.utils.rnn import pad_sequence

def multimodal_collate(batch):
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    vision = torch.stack([item[2] for item in batch])
    audio = torch.stack([item[3] for item in batch])
    labels = torch.tensor([item[4] for item in batch], dtype=torch.long)
    raw_texts = [item[5] for item in batch]

    return input_ids_padded, attention_mask_padded, vision, audio, labels, raw_texts
