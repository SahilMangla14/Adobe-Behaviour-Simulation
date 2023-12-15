import pandas as pd
import numpy as np
import spacy
import torch
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import PCA
from datetime import datetime
import os
import ast 
import warnings


def bert_embeddings_apply_pca(df, content_column, output_csv_path, threshold=0.95):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Function to get BERT embeddings for a given text
    def get_bert_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = inputs.to(device)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.cpu().mean(dim=1).squeeze().detach().numpy()
        return embeddings

    # Apply BERT embeddings to the specified content column
    df['bert_embeddings'] = df[content_column].apply(get_bert_embeddings)

    # Extract BERT embeddings as a numpy array
    bert_embeddings_array = np.stack(df['bert_embeddings'].to_numpy())

    # Apply PCA to reduce the dimension to the optimal number of components
    optimal_components = 356
    pca = PCA(n_components=optimal_components)
    bert_embeddings_reduced = pca.fit_transform(bert_embeddings_array)

    # Create a new DataFrame for the reduced embeddings
    df_reduced_embeddings = pd.DataFrame(bert_embeddings_reduced, columns=[f'component_{i+1}' for i in range(optimal_components)])

    # Combine 'id', 'content', and reduced embeddings into a new DataFrame
    df_combined = pd.concat([df[['id', content_column]], df_reduced_embeddings], axis=1)

    # Save the new DataFrame to a CSV file
    # df_combined.to_csv(output_csv_path, index=False)

    print(f'Optimal number of components: {optimal_components}')
    
    return df_combined


def convert_to_epoch(datetime_str):
    dt_object = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    return int(dt_object.timestamp())



if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    current_directory = os.getcwd()
    
    # COMPANY TEST DATASET
    # df = pd.read_csv('./data_company/behaviour_simulation_test_company_companyrank_usernamerank.csv')
    
    # print("TIME...")
    # df['epoch_time'] = df['date'].apply(lambda x: convert_to_epoch(x))
    
    # # PCA components content bert embeddings
    # print("PCA...")
    # PCA_reduced_embeddings_content =  bert_embeddings_apply_pca(df,'content','.')
    
    # pca_path = os.path.join(current_directory,'data_company' ,'test_company_pca.csv')
    # PCA_reduced_embeddings_content.to_csv(pca_path, index = False)
    
    # PCA_reduced_embeddings_content = PCA_reduced_embeddings_content.drop(columns = ['id','content'])
    
    # print("CONCATENATION...")
    # df_company_rank = df['company_rank']
    # df_username_rank = df['username_rank']
    # df_epoch_time = df['epoch_time']
    
    # df_final = pd.concat([PCA_reduced_embeddings_content,df_username_rank,df_epoch_time,df_company_rank], axis = 1)
    
    # path = os.path.join(current_directory,'data_company' ,'df_final.csv')
    # df_final.to_csv(path,index = False)
    # print("DONE HURRAY!")



    # TIME TEST DATASEST
    df = pd.read_csv('./data_time/behaviour_simulation_test_time_companyrank_usernamerank.csv')
    
    print("TIME...")
    df['epoch_time'] = df['date'].apply(lambda x: convert_to_epoch(x))
    
    # PCA components content bert embeddings
    print("PCA...")
    PCA_reduced_embeddings_content =  bert_embeddings_apply_pca(df,'content','.')
    
    pca_path = os.path.join(current_directory,'data_time' ,'test_company_pca.csv')
    PCA_reduced_embeddings_content.to_csv(pca_path, index = False)
    
    PCA_reduced_embeddings_content = PCA_reduced_embeddings_content.drop(columns = ['id','content'])
    
    print("CONCATENATION...")
    df_company_rank = df['company_rank']
    df_username_rank = df['username_rank']
    df_epoch_time = df['epoch_time']
    
    df_final = pd.concat([PCA_reduced_embeddings_content,df_username_rank,df_epoch_time,df_company_rank], axis = 1)
    
    path = os.path.join(current_directory,'data_time' ,'df_final.csv')
    df_final.to_csv(path,index = False)
    print("DONE HURRAY!")
