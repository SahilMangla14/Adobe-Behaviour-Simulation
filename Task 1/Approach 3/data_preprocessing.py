import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
    
    def map_to_one(self):
        self.df['likes'].replace(0, 1, inplace=True)    
        
    def remove_outliers(self):
        self.df['new_feature'] = (self.df['company_rank'] + self.df['username_rank']) / self.df['likes']
        print(len(self.df))
        self.df = self.df.dropna()
        print(len(self.df))
        # Find the standard deviation of 'new_feature'
        std_dev = self.df['new_feature'].std()

        # Define a z-score threshold for outliers (e.g., 3 standard deviations)
        z_score_threshold = 3

        # Calculate z-scores for 'new_feature'
        z_scores = np.abs((self.df['new_feature'] - self.df['new_feature'].mean()) / std_dev)

        # Remove rows with z-scores above the threshold
        self.df = self.df[z_scores <= z_score_threshold]
        self.df = self.df.drop(columns = ['new_feature'])
        
        

    def scale_target_variable(self):
        scaler_likes = MinMaxScaler()
        self.df['likes'] = scaler_likes.fit_transform(self.df['likes'].values.reshape(-1, 1))
        self.scaler_likes = scaler_likes

    def scale_features(self):
        # features_to_scale = ['epoch_time', 'total_rank_entity', 'company_rank']
        features_to_scale = ['epoch_time', 'username_rank', 'company_rank']
        for index, feature in enumerate(features_to_scale):
            scaler = MinMaxScaler()
            self.df[feature] = scaler.fit_transform(self.df[feature].values.reshape(-1, 1))
            if (index == 0):
                self.scaler_epoch_time = scaler
            elif(index == 1):
                self.scaler_username_rank = scaler 
            else:
                self.scaler_company_rank = scaler

    def drop_na_rows(self):
        self.df = self.df.dropna()

    def split_data(self, test_size=0.2, random_state=0):
        X = self.df.drop(columns='likes')
        y = self.df['likes']
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def save_scalers(self, scaler_file_path, scaler_epoch_time_file_path, scaler_username_rank_file_path, scaler_company_rank_file_path):
        self.scaler_likes_file_path = scaler_file_path
        self.scaler_epoch_time_file_path = scaler_epoch_time_file_path
        self.scaler_username_rank_file_path = scaler_username_rank_file_path
        self.scaler_company_rank_file_path = scaler_company_rank_file_path

        joblib.dump(self.scaler_likes, scaler_file_path)
        joblib.dump(self.scaler_epoch_time, scaler_epoch_time_file_path)
        joblib.dump(self.scaler_username_rank, scaler_username_rank_file_path)
        joblib.dump(self.scaler_company_rank, scaler_company_rank_file_path)

    def load_scalers(self, scaler_file_path, scaler_epoch_time_file_path, scaler_username_rank_file_path, scaler_company_rank_file_path):
        self.scaler_likes = joblib.load(scaler_file_path)
        self.scaler_epoch_time = joblib.load(scaler_epoch_time_file_path)
        self.scaler_username_rank = joblib.load(scaler_username_rank_file_path)
        self.scaler_company_rank = joblib.load(scaler_company_rank_file_path)
