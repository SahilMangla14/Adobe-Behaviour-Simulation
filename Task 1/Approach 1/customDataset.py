from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import torch

class CustomDataset(Dataset):
    def __init__(self, tweets, likes, tokenizer, bert_model):
        self.tweets = tweets
        self.likes = likes
        self.tokenizer = tokenizer
        self.bert_model = bert_model

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        like = torch.tensor(self.likes[idx], dtype=torch.float32)  # Convert to PyTorch tensor

        # Tokenize and get BERT embeddings
        inputs = self.tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=128)
        input_ids = inputs['input_ids'].squeeze()  # Remove the added batch dimension
        attention_mask = inputs['attention_mask'].squeeze()  # Remove the added batch dimension

        outputs = self.bert_model(**inputs)
        pooled_output = outputs['pooler_output'].detach().numpy()

        return input_ids, attention_mask, like
