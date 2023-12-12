import torch.nn as nn
from transformers import BertModel

class RegressionHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionHead, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BERTRegressionModel(nn.Module):
    def __init__(self, bert_model, regression_head_date, regression_head_content, regression_head_username, regression_head_company):
        super(BERTRegressionModel, self).__init__()
        self.bert_model = bert_model
        self.regression_head_date = regression_head_date
        self.regression_head_content = regression_head_content
        self.regression_head_username = regression_head_username
        # self.regression_head_media = regression_head_media
        self.regression_head_company = regression_head_company

    def forward(self, input_ids_date, attention_mask_date, input_ids_content, attention_mask_content, input_ids_username, attention_mask_username, input_ids_company, attention_mask_company):
        # Embeddings for date
        outputs_date = self.bert_model(input_ids=input_ids_date, attention_mask=attention_mask_date)
        pooled_output_date = outputs_date['pooler_output']
        regression_output_date = self.regression_head_date(pooled_output_date)

        # Embeddings for content
        outputs_content = self.bert_model(input_ids=input_ids_content, attention_mask=attention_mask_content)
        pooled_output_content = outputs_content['pooler_output']
        regression_output_content = self.regression_head_content(pooled_output_content)

        # Embeddings for username
        outputs_username = self.bert_model(input_ids=input_ids_username, attention_mask=attention_mask_username)
        pooled_output_username = outputs_username['pooler_output']
        regression_output_username = self.regression_head_username(pooled_output_username)

        # Embeddings for media
        # outputs_media = self.bert_model(input_ids=input_ids_media, attention_mask=attention_mask_media)
        # pooled_output_media = outputs_media['pooler_output']
        # regression_output_media = self.regression_head_media(pooled_output_media)

        # Embeddings for company
        outputs_company = self.bert_model(input_ids=input_ids_company, attention_mask=attention_mask_company)
        pooled_output_company = outputs_company['pooler_output']
        regression_output_company = self.regression_head_company(pooled_output_company)

        # Combine the outputs with learned weights
        combined_output = regression_output_date + regression_output_content + regression_output_username + regression_output_company

        return combined_output

