import pandas as pd
import numpy as np
import spacy
import torch
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
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

    # Apply PCA to reduce the dimension to the optimal number of components (300)
    optimal_components = 356
    pca = PCA(n_components=optimal_components)
    bert_embeddings_reduced = pca.fit_transform(bert_embeddings_array)

    # Create a new DataFrame for the reduced embeddings
    df_reduced_embeddings = pd.DataFrame(bert_embeddings_reduced, columns=[f'component_{i+1}' for i in range(optimal_components)])

    # Save the DataFrame with reduced embeddings to a CSV file
    # df_reduced_embeddings.to_csv(output_csv_path, index=False)

    # return df_reduced_embeddings
    # Combine 'id', 'content', and reduced embeddings into a new DataFrame
    df_combined = pd.concat([df[['id', content_column]], df_reduced_embeddings], axis=1)

    # Save the new DataFrame to a CSV file
    # df_combined.to_csv(output_csv_path, index=False)

    print(f'Optimal number of components: {optimal_components}')
    
    return df_combined


def convert_to_epoch(datetime_str):
    dt_object = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    return int(dt_object.timestamp())

def get_bert_embedding(text):
        tokens = tokenizer(text, return_tensors="pt")
        tokens = tokens.to(device)
        with torch.no_grad():
            output = model(**tokens)
        return output.last_hidden_state.cpu().mean(dim=1).squeeze().numpy()


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
    
    # Take input from the user for each field
    # id_val = int(input("Enter ID: "))
    # date_val = input("Enter Date (in the format 'YYYY-MM-DD HH:mm:ss'): ")
    # content_val = input("Enter Content: ")
    # username_val = input("Enter Username: ")
    # media_val = input("Enter Media (as a string): ")
    # company_val = input("Enter Inferred Company: ")
    
    # id_val = 1
    # date_val = "2021-11-08 00:40:49"
    # content_val = "Andres, a Pharmacy Manager at Store 4669 in Pinellas Park, FL, recently helped save the lives of TWO customers in TWO days! One day he was administering life-saving medication &amp; the next day his quick thinking had his tech, Shay, calling 911 🚨 in the nick of time 🙌 <hyperlink>"
    # username_val = "walmartworld"
    # media_val = "[Photo(previewUrl='https://pbs.twimg.com/media/FDocI4fXsAcP-kl?format=jpg&name=small', fullUrl='https://pbs.twimg.com/media/FDocI4fXsAcP-kl?format=jpg&name=large')]"
    # company_val = "walmart"

    # # Create a dictionary with the input values
    # data = {
    #     'id': [id_val],
    #     'date': [date_val],
    #     'content': [content_val],
    #     'username': [username_val],
    #     'media': [media_val],
    #     'inferred company': [company_val]
    # }

    # # # Create a Pandas DataFrame
    # df2 = pd.DataFrame(data)
    
    adobe_path = './custom/adobe_test.csv'
    df_test_adobe = pd.read_csv(adobe_path)
    
    df3 = pd.read_csv('./data_time/behaviour_simulation_test_time.csv').head(500)
    
    df_final_test = pd.concat([df_test_adobe, df3], ignore_index=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # READ company_analysis.csv and COMPANY RANK
    df1 = pd.read_csv('./data_train/company_analysis.csv')

    # Step 1: Find unique companies in the second dataframe
    df2 = df_final_test.copy()
    unique_companies_df2 = df2['inferred company'].unique()
    print("COMPANY BERT EMBEDDINGS...")
    df1['embedding'] = df1['inferred company'].apply(get_bert_embedding)

    # Step 3: Create a map for companies in the second dataframe
    company_rank_map = {}
    print("COMPANY SCORE...")

    for company_df2 in unique_companies_df2:
        if company_df2 in df1['inferred company'].values:
            rank = df1[df1['inferred company'] == company_df2]['weighted_rank'].values[0]
        else:
            # Find BERT embedding for the current unique company in df2
            embedding_df2 = get_bert_embedding(company_df2)

            # Find cosine similarity with companies in the first dataframe
            similarities = cosine_similarity([embedding_df2], df1['embedding'].tolist())[0]
            most_similar_company = df1.loc[similarities.argmax(), 'inferred company']
            rank = df1[df1['inferred company'] == most_similar_company]['weighted_rank'].values[0]

        company_rank_map[company_df2] = rank

    # Step 4: Create a new column in the second dataframe with the company ranks
    df2['company_rank'] = df2['inferred company'].map(company_rank_map)


    # READ username_analysis.csv and USERNAME RANK
    df1 = pd.read_csv('./data_train/username_analysis.csv')
    
    
    print("USERNAME EMBEDDINGS...")
    df1['embedding'] = df1['username'].apply(get_bert_embedding)
    unique_username_df2 = df2['username'].unique()

    print("USERNAME SCORE...")
    username_rank_map = {}

    for username_df2 in unique_username_df2:
        if username_df2 in df1['username'].values:
            rank = df1[df1['username'] == username_df2]['weighted_rank'].values[0]
        else:
            # Find BERT embedding for the current unique company in df2
            embedding_df2 = get_bert_embedding(username_df2)

            # Find cosine similarity with companies in the first dataframe
            similarities = cosine_similarity([embedding_df2], df1['embedding'].tolist())[0]
            most_similar_username = df1.loc[similarities.argmax(), 'username']
            rank = df1[df1['username'] == most_similar_username]['weighted_rank'].values[0]

        username_rank_map[username_df2] = rank
    
    df2['username_rank'] = df2['username'].map(username_rank_map)
    
    df2.to_csv('./custom/behaviour_simulation_companyrank_usernamerank.csv')
    df = df2
    
    print("NER Entities and K Means")

    # TIME TEST DATASEST
    # df = pd.read_csv('./data_time/behaviour_simulation_test_time_companyrank_usernamerank.csv')
    
    print("TIME...")
    df['epoch_time'] = df['date'].apply(lambda x: convert_to_epoch(x))
    
    # PCA components content bert embeddings
    print("PCA...")
    PCA_reduced_embeddings_content =  bert_embeddings_apply_pca(df,'content','.')
    
    # pca_path = os.path.join(current_directory,'data_time' ,'test_company_pca.csv')
    pca_path = os.path.join(current_directory,'custom' ,'pca.csv')
    PCA_reduced_embeddings_content.to_csv(pca_path, index = False)
    
    PCA_reduced_embeddings_content = PCA_reduced_embeddings_content.drop(columns = ['id','content'])
    
    print("CONCATENATION...")
    df_company_rank = df['company_rank']
    df_username_rank = df['username_rank']
    df_epoch_time = df['epoch_time']
    
    df_final = pd.concat([PCA_reduced_embeddings_content,df_username_rank,df_epoch_time,df_company_rank], axis = 1)
    
    # path = os.path.join(current_directory,'data_time' ,'df_final.csv')
    df_final = df_final.head(len(df_test_adobe))
    path = os.path.join(current_directory,'custom' ,'df_final.csv')
    df_final.to_csv(path,index = False)
    print("DONE HURRAY!")
    
    # PCA_reduced_embeddings_content =  bert_embeddings_apply_pca(df2,'content','.')