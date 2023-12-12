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
        x = self.fc2(x)
        return x


class BERTRegressionModel(nn.Module):
    def __init__(self, bert_model, regression_head):
        super(BERTRegressionModel, self).__init__()
        self.bert_model = bert_model
        self.regression_head = regression_head

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        regression_output = self.regression_head(pooled_output)
        return regression_output