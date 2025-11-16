import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class TextFeatureExtractor(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", output_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.conv = nn.Conv1d(in_channels=768, out_channels=output_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        x = x.transpose(1, 2)

        x = self.conv(x)
        x = self.pool(x).squeeze(-1)

        return x
