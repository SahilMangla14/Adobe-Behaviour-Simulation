import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from customDataset import CustomDataset
from customBertRegression import BERTRegressionModel, RegressionHead
from train import train_model
from helpers import collate_fn, evaluate_model, predict_model


def main():
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv('./behaviour_simulation_train.csv')
    df['ner_entities'] 
        
    # Create dataset instance
    dataset = CustomDataset(tweet_desc, tweet_likes)

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Create data loaders
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize the model and optimizer
    input_size = 768  # Size of BERT's hidden layers
    hidden_size = 256  # Adjust based on your requirements
    output_size = 1  # For predicting a single numerical value

    regression_head = RegressionHead(input_size, hidden_size, output_size)
    model = BERTRegressionModel(bert_model, regression_head)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)

    # Train the model
    num_epochs = 30
    train_model(model, train_dataloader, criterion, optimizer, num_epochs, device)

    # Evaluate the model
    evaluate_model(model, test_dataloader, criterion, device)

    # Predict using the model and save results
    comparison_df = predict_model(model, test_dataloader, scaler_likes, device)


if __name__ == "__main__":
    main()