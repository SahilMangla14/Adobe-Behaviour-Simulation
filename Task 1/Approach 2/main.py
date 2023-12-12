import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.nn.utils.rnn import pad_sequence
from customDataset import CustomDataset
from customBertRegression import BERTRegressionModel, RegressionHead
from train import train_model
from helpers import collate_fn, evaluate_model, predict_model

def main():
    
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Create dataset and split into training and testing sets
    dataset = CustomDataset(tweet_date, tweet_content, tweet_username, tweet_media, tweet_company, tweet_likes)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Create data loaders
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model parameters
    input_size = 768  # Size of BERT's hidden layers
    hidden_size = 256  # Adjust based on your requirements
    output_size = 1  # For predicting a single numerical value

    # Regression head
    regression_head = RegressionHead(input_size, hidden_size, output_size)

    # Model
    model = BERTRegressionModel(bert_model, regression_head)

    # Training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    num_epochs = 30
    train(model, train_dataloader, criterion, optimizer, device, num_epochs)
    
    torch.save(model,'./regression_model.pth')



if __name__ == "__main__":
    main()