import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.nn.utils.rnn import pad_sequence
from customDataset import CustomDataset
from RegressionHead_Bert import BERTRegressionModel, RegressionHead
from train import train_model
from helpers import get_bert_embeddings , extract_topics, process_row, collate_fn, evaluate_model, predict_model
from transformers import pipeline
import spacy
from tqdm import tqdm
import pandas as pd 
import numpy as np

def main():
    
    # Load the NER model
    ner = spacy.load("en_core_web_sm")
    
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    df = pd.read_csv('./behaviour_simulation_train.csv')
    df = df.head(20)
    
    scaler_likes = MinMaxScaler()
    df['likes'] = scaler_likes.fit_transform(df['likes'].values.reshape(-1, 1))
        
    # Apply the function to the DataFrame
    tqdm.pandas()  # Enable progress_apply, optional
    df['content_ner_entities'] = df['content'].progress_apply(lambda text: str(process_row(ner, {'content': text})))
    
    tweet_date = df['date']
    tweet_content = df['content_ner_entities']
    tweet_username = df['username']
    tweet_company = df['inferred company']
    tweet_likes = df['likes']
    
    
    # Create dataset and split into training and testing sets
    dataset = CustomDataset(tweet_date, tweet_content, tweet_username, tweet_company, tweet_likes, tokenizer, bert_model)
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
    model = BERTRegressionModel(bert_model, regression_head, regression_head, regression_head, regression_head)

    # Training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    num_epochs = 30
    # print("ON TRAINING")
    train_model(model, train_dataloader, criterion, optimizer, device, num_epochs)
    
    torch.save(model,'./regression_model.pth')
    
    # Evaluate the model
    evaluate_model(model, test_dataloader, criterion, device)

    # Predict using the model and save results
    comparison_df = predict_model(model, test_dataloader, scaler_likes, device)



if __name__ == "__main__":
    main()
