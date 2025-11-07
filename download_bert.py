from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
model = BertModel.from_pretrained("./bert-base-uncased")
