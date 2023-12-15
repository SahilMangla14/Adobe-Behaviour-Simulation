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

def calculate_company_weighted_rank(df):
    # Group by 'company' and sum the 'likes'
    company_likes = df.groupby('inferred company')['likes'].sum().reset_index()
    sorted_company_likes = company_likes.sort_values(by='likes', ascending=False)
    
    # Assign ranks to companies based on likes
    sorted_company_likes['rank'] = sorted_company_likes['likes'].rank(ascending=True, method='min')

    # Define multiple thresholds and corresponding factors
    thresholds_factors = [
        (20000000, 6),
        (10000000, 5.5),
        (5000000, 5),
        (2000000, 4.5),
        (1000000, 4),
        (500000, 3.5),
        (100000, 3.0),
        (50000, 2.5),
        (10000, 2),
        (5000, 1.5)
    ]

    # Define a function to apply the appropriate factor based on likes
    def apply_factor(row):
        for threshold, factor in thresholds_factors:
            if row['likes'] > threshold:
                return row['rank'] * factor
        return row['rank']

    # Apply the function to create the 'weighted_rank' column
    sorted_company_likes['weighted_rank'] = sorted_company_likes.apply(apply_factor, axis=1)
    
    return sorted_company_likes


def calculate_username_weighted_rank(df):
    # Group by 'company' and sum the 'likes'
    username_likes = df.groupby('username')['likes'].sum().reset_index()
    sorted_username_likes = username_likes.sort_values(by='likes', ascending=False)
    
    # Assign ranks to companies based on likes
    sorted_username_likes['rank'] = sorted_username_likes['likes'].rank(ascending=True, method='min')

    # # Define multiple thresholds and corresponding factors
    thresholds_factors = [
        (20000000, 6),
        (10000000, 5.5),
        (5000000, 5),
        (2000000, 4.5),
        (1000000, 4),
        (500000, 3.5),
        (100000, 3.0),
        (50000, 2.5),
        (10000, 2),
        (5000, 1.5)
    ]

    # Define a function to apply the appropriate factor based on likes
    def apply_factor(row):
        for threshold, factor in thresholds_factors:
            if row['likes'] > threshold:
                return row['rank'] * factor
        return row['rank']

    # Apply the function to create the 'weighted_rank' column
    sorted_username_likes['weighted_rank'] = sorted_username_likes.apply(apply_factor, axis=1)
    
    return sorted_username_likes


def analyze_ner_entities(df):
    # Explode the 'ner_entities' list into separate rows
    df_exploded = df.explode('ner_entities')

    # Convert 'ner_entities' column to a list of literals
    if(isinstance(df['ner_entities'].iloc[0], str)):
        df['ner_entities'] = df['ner_entities'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

    # Create a new DataFrame to store the results
    ner_entities_likes_data = []

    # Iterate over rows and add individual ner_entities with their corresponding likes and count
    for index, row in df.iterrows():
        ner_entities = row['ner_entities']
        likes = row['likes']

        for ner_entity in ner_entities:
            ner_entities_likes_data.append({'ner_entity': ner_entity, 'likes': likes, 'count': 1})

    # Convert the list of dictionaries to a DataFrame
    ner_entities_likes = pd.DataFrame(ner_entities_likes_data)

    # Group by 'ner_entity' and calculate sum of 'likes' and count of occurrences
    grouped_ner_entities_likes = ner_entities_likes.groupby('ner_entity').agg({'likes': 'sum', 'count': 'count'}).reset_index()

    # Calculate average likes for each entity
    grouped_ner_entities_likes['avg_likes'] = grouped_ner_entities_likes['likes'] / grouped_ner_entities_likes['count']

    # Sort the DataFrame based on 'likes' in descending order
    sorted_ner_entities_likes = grouped_ner_entities_likes.sort_values(by='avg_likes', ascending=False)

    # Assign ranks to ner_entities based on likes (bigger number for more likes)
    sorted_ner_entities_likes['rank'] = sorted_ner_entities_likes['likes'].rank(ascending=True, method='min')

    
    thresholds_factors = [
        (15000, 2.5),
        (10000, 2.0),
        (5000, 1.5)
    ]

    # Define a function to apply the appropriate factor based on likes
    def apply_factor_entity(row):
        for threshold, factor in thresholds_factors:
            if row['avg_likes'] > threshold:
                return row['rank'] * factor
        return row['rank']

    # Apply the function to create the 'weighted_rank' column
    sorted_ner_entities_likes['weighted_rank'] = sorted_ner_entities_likes.apply(apply_factor_entity, axis=1)
    
    # Save the result to a CSV file
    # sorted_ner_entities_likes.to_csv('/content/drive/MyDrive/Adobe/entity_analysis.csv', index=False)

    return sorted_ner_entities_likes



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

    # Apply PCA for different numbers of components
    pca = PCA()
    pca.fit(bert_embeddings_array)

    # Plot explained variance as a function of the number of components (optional)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # Find the number of components that capture the specified threshold of the variance
    optimal_components = np.argmax(cumulative_explained_variance >= threshold) + 1

    # Apply PCA to reduce the dimension to the optimal number of components
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

    
def OneHotEncoding_username(df):
    df_encoded = pd.get_dummies(df, columns=['username'], prefix='username')
    df[df_encoded.columns] = df_encoded
    
def extract_topics(ner, text):
    doc = ner(text)
    entities = [ent.text for ent in doc.ents]

    return entities

def process_row(ner, row):
    text = row['content']
    ner_entities = extract_topics(ner, text)
    return ner_entities


def calculate_total_rank(row, entity_rank_map):
    ner_entities = row['ner_entities']
    
    if not ner_entities:
        return 0
    
    return max(entity_rank_map.get(entity, 0) for entity in ner_entities)
   


if __name__ == "__main__":
    # Filter out warnings
    warnings.filterwarnings("ignore")
    
    df = pd.read_csv('./ner_entity_300k_updated.csv')
    if(isinstance(df['likes'].iloc[0], str) == True):
        df['likes'] = df['likes'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else 0)
    
    # Get the current working directory
    current_directory = os.getcwd()
    
    # Assign company rank
    print("COMPANY RANK...")
    company_analysis = calculate_company_weighted_rank(df)
    df['company_rank'] = df['inferred company'].map(company_analysis.set_index('inferred company')['weighted_rank'])
    
    
    print("USERNAME RANK...")
    username_analysis = calculate_username_weighted_rank(df)
    df['username_rank'] = df['username'].map(username_analysis.set_index('username')['weighted_rank'])
    
    
    # df_cr_path = os.path.join(current_directory, 'df_company_rank.csv')
    # df.to_csv(df_cr_path, index = False)
    
    # Find ner entities and then assign total rank entity
    # ner = spacy.load("en_core_web_sm")
    # df['ner_entities'] = df['content'].apply(lambda text: [ent.text for ent in ner(text).ents])
    # print("ENTITY RANK...")
    # entity_analysis = analyze_ner_entities(df)
    # entity_rank_map = dict(zip(entity_analysis['ner_entity'], entity_analysis['weighted_rank']))
    # df['total_rank_entity'] = df.apply(calculate_total_rank, entity_rank_map=entity_rank_map, axis=1)
    
    
    # df_ter_path = os.path.join(current_directory, 'df_entity_rank.csv')
    # df.to_csv(df_ter_path, index = False)
    
    # EPOCH TIME AND USERNAME ONE HOT ENCODING
    # df = pd.read_csv('./df_entity_rank.csv')
    print("TIME...")
    df['epoch_time'] = df['date'].apply(lambda x: convert_to_epoch(x))
    
    # print("ONE HOT ENCODING...")
    # OneHotEncoding_username(df)
    
    # PCA components content bert embeddings
    print("PCA...")
    # PCA_reduced_embeddings_content =  bert_embeddings_apply_pca(df,'content','.')
    
    # pca_path = os.path.join(current_directory, 'pca.csv')
    # PCA_reduced_embeddings_content.to_csv(pca_path, index = False)
    
    PCA_reduced_embeddings_content = pd.read_csv('./pca.csv')
    PCA_reduced_embeddings_content = PCA_reduced_embeddings_content.drop(columns = ['id','content'])

    
    # print("CONCATENATION...")
    # df_likes = df['likes']
    # df_company_rank = df['company_rank']
    # df_total_entity_rank = df['total_rank_entity']
    # df_epoch_time = df['epoch_time']
    # df_username_onehot = df.drop(columns=['id','date','likes','content','username','media','inferred company','company_rank','ner_entities','total_rank_entity','epoch_time'])
    
    # df_train = pd.concat([PCA_reduced_embeddings_content,df_username_onehot,df_epoch_time,df_total_entity_rank,df_company_rank,df_likes], axis = 1)
    # df_all = pd.concat([df,PCA_reduced_embeddings_content], axis = 1)
    
    
    # # Specify the CSV file path
    # df_train_path = os.path.join(current_directory, 'df_train2.csv')
    # df_train.to_csv(df_train_path, index = False)
    
    # df_all_path = os.path.join(current_directory, 'df_all.csv')
    # df_all.to_csv(df_all_path, index = False)
    # print("DONE. HURRAY")
    
    
    print("CONCATENATION...")
    df_likes = df['likes']
    df_company_rank = df['company_rank']
    df_epoch_time = df['epoch_time']
    df_username_rank = df['username_rank']
    
    df_train = pd.concat([PCA_reduced_embeddings_content,df_username_rank,df_epoch_time,df_company_rank, df_likes], axis = 1)
    
    path = os.path.join(current_directory,'data_train' ,'df_train.csv')
    df_train.to_csv(path,index = False)
    print("DONE HURRAY!")
    
    
    