import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from customDataset import CustomDataset
from customBertRegression import BERTRegressionModel, RegressionHead
from train import train_model
from helpers import get_bert_embeddings , extract_topics, process_row, make_tweetDesc_likes , collate_fn, evaluate_model, predict_model
from transformers import pipeline
import spacy
from tqdm import tqdm

def main():
    
    # Load the NER model
    ner = spacy.load("en_core_web_sm")
    
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv('./behaviour_simulation_train.csv')
    df = df.head(20)
    
    scaler_likes = MinMaxScaler()
    df['likes'] = scaler_likes.fit_transform(df['likes'].values.reshape(-1, 1))
        
    # Apply the function to the DataFrame
    tqdm.pandas()  # Enable progress_apply, optional
    df['company_ner_entities'] = df['content'].progress_apply(lambda text: process_row(ner, {'content': text}))

    tweet_desc, tweet_likes = make_tweetDesc_likes(df)
    
    # Create dataset instance
    dataset = CustomDataset(tweet_desc, tweet_likes, tokenizer, bert_model)
    

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
    print("ON TRAINING")
    num_epochs = 30
    train_model(model, train_dataloader, criterion, optimizer, num_epochs, device)

    torch.save(model,'./regression_model.pth')
    
    # Evaluate the model
    evaluate_model(model, test_dataloader, criterion, device)

    # Predict using the model and save results
    comparison_df = predict_model(model, test_dataloader, scaler_likes, device)


if __name__ == "__main__":
    main()
