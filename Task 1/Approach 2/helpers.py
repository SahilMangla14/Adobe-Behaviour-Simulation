from sklearn.metrics import mean_squared_error
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import torch

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = inputs.to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.cpu().mean(dim=1).squeeze().detach().numpy()
    return embeddings

def extract_topics(ner, text):
    doc = ner(text)
    entities = [ent.text for ent in doc.ents]

    return entities

def process_row(ner, row):
    text = row['content']
    ner_entities = extract_topics(ner, text)
    return ner_entities

def collate_fn(batch):
    date_input_ids, date_attention_mask , content_input_ids, content_attention_mask, username_input_ids, username_attention_mask, company_input_ids, company_attention_mask, labels = zip(*batch)

    # Pad sequences to the maximum length in the batch
    date_input_ids = pad_sequence(date_input_ids, batch_first=True)
    date_attention_mask = pad_sequence(date_attention_mask, batch_first=True)

    content_input_ids = pad_sequence(content_input_ids, batch_first=True)
    content_attention_mask = pad_sequence(content_attention_mask, batch_first=True)

    username_input_ids = pad_sequence(username_input_ids, batch_first=True)
    username_attention_mask = pad_sequence(username_attention_mask, batch_first=True)

    # media_input_ids = pad_sequence(media_input_ids, batch_first=True)
    # media_attention_mask = pad_sequence(media_attention_mask, batch_first=True)

    company_input_ids = pad_sequence(company_input_ids, batch_first=True)
    company_attention_mask = pad_sequence(company_attention_mask, batch_first=True)

    return  date_input_ids, date_attention_mask , content_input_ids, content_attention_mask, username_input_ids, username_attention_mask, company_input_ids, company_attention_mask, torch.stack(labels)

def evaluate_model(model, test_dataloader, criterion, device):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            date_input_ids, date_attention_mask, content_input_ids, content_attention_mask, username_input_ids, username_attention_mask, company_input_ids, company_attention_mask ,labels = batch

            date_input_ids, date_attention_mask, labels = date_input_ids.to(device), date_attention_mask.to(device), labels.to(device)
            content_input_ids, content_attention_mask = content_input_ids.to(device), content_attention_mask.to(device)
            username_input_ids, username_attention_mask = username_input_ids.to(device), username_attention_mask.to(device)
            # media_input_ids, media_attention_mask = media_input_ids.to(device), media_attention_mask.to(device)
            company_input_ids, company_attention_mask = company_input_ids.to(device), company_attention_mask.to(device)
            
            predictions = model(date_input_ids, date_attention_mask, content_input_ids, content_attention_mask, username_input_ids, username_attention_mask, company_input_ids, company_attention_mask)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    mse = mean_squared_error(all_labels, all_predictions)
    rmse = mean_squared_error(all_labels, all_predictions, squared = False)
    print(f'Mean Squared Error on Test Set: {mse:.4f}')
    print(f'Root Mean Squared Error on Test Set: {rmse:.4f}')
    return mse, rmse


def predict_model(model, test_dataloader, scaler_likes, device):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            date_input_ids, date_attention_mask, content_input_ids, content_attention_mask, username_input_ids, username_attention_mask, company_input_ids, company_attention_mask ,labels = batch

            date_input_ids, date_attention_mask, labels = date_input_ids.to(device), date_attention_mask.to(device), labels.to(device)
            content_input_ids, content_attention_mask = content_input_ids.to(device), content_attention_mask.to(device)
            username_input_ids, username_attention_mask = username_input_ids.to(device), username_attention_mask.to(device)
            # media_input_ids, media_attention_mask = media_input_ids.to(device), media_attention_mask.to(device)
            company_input_ids, company_attention_mask = company_input_ids.to(device), company_attention_mask.to(device)
            
            predictions = model(date_input_ids, date_attention_mask, content_input_ids, content_attention_mask, username_input_ids, username_attention_mask, company_input_ids, company_attention_mask)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert tensors to NumPy arrays
    y_test_np = np.array(all_labels)
    y_pred_np = np.array(all_predictions)

    # Inverse transform using the scaler
    y_test_inv = scaler_likes.inverse_transform(y_test_np.reshape(-1,1))
    y_pred_inv = scaler_likes.inverse_transform(y_pred_np.reshape(-1,1))

    # Stack the arrays side by side
    comparison_result = np.column_stack((y_test_inv, y_pred_inv))

    # Create a DataFrame for better visualization
    comparison_df = pd.DataFrame(comparison_result, columns=['y_test', 'y_pred'])

    # Display the DataFrame
    comparison_df.to_csv('./ann_results.csv', index=False)

    return comparison_df



