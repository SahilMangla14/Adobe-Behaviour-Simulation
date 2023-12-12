from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import torch

class CustomDataset(Dataset):
    def __init__(self, tweet_date, tweet_content, tweet_username, tweet_company, likes, tokenizer, bert_model):
        self.tweet_date = tweet_date
        self.tweet_content = tweet_content
        self.tweet_username = tweet_username
        # self.tweet_media = tweet_media
        self.tweet_company =tweet_company
        self.likes = likes
        self.tokenizer = tokenizer
        self.model = bert_model

    def __len__(self):
        return len(self.tweet_date)

    def __getitem__(self, idx):
        date = self.tweet_date[idx]
        content = self.tweet_content[idx]
        username = self.tweet_username[idx]
        # media = self.tweet_media[idx]
        company = self.tweet_company[idx]
        like = torch.tensor(self.likes[idx], dtype=torch.float32)  # Convert to PyTorch tensor

        # Tokenize and get BERT embeddings for each feature
        date_inputs = self.tokenizer(date, return_tensors="pt", truncation=True, padding=True, max_length=128)
        date_input_ids = date_inputs['input_ids'].squeeze()
        date_attention_mask = date_inputs['attention_mask'].squeeze()

        content_inputs = self.tokenizer(content, return_tensors="pt", truncation=True, padding=True, max_length=128)
        content_input_ids = content_inputs['input_ids'].squeeze()  # Remove the added batch dimension
        content_attention_mask = content_inputs['attention_mask'].squeeze()  # Remove the added batch dimension

        username_inputs = self.tokenizer(username, return_tensors="pt", truncation=True, padding=True, max_length=128)
        username_input_ids = username_inputs['input_ids'].squeeze()
        username_attention_mask = username_inputs['attention_mask'].squeeze()

        # media_inputs = tokenizer(media, return_tensors="pt", truncation=True, padding=True, max_length=128)
        # media_input_ids = media_inputs['input_ids'].squeeze()
        # media_attention_mask = media_inputs['attention_mask'].squeeze()

        company_inputs = self.tokenizer(company, return_tensors="pt", truncation=True, padding=True, max_length=128)
        company_input_ids = company_inputs['input_ids'].squeeze()
        company_attention_mask = company_inputs['attention_mask'].squeeze()


        return date_input_ids, date_attention_mask , content_input_ids, content_attention_mask, username_input_ids, username_attention_mask, company_input_ids, company_attention_mask, like
