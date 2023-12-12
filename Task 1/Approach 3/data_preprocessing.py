import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def scale_target_variable(self):
        scaler_likes = MinMaxScaler()
        self.df['likes'] = scaler_likes.fit_transform(self.df['likes'].values.reshape(-1, 1))
        self.scaler_likes = scaler_likes

    def scale_features(self):
        features_to_scale = ['epoch_time', 'total_rank_entity', 'company_rank']
        for feature in features_to_scale:
            scaler = MinMaxScaler()
            self.df[feature] = scaler.fit_transform(self.df[feature].values.reshape(-1, 1))

    def drop_na_rows(self):
        self.df = self.df.dropna()

    def split_data(self, test_size=0.2, random_state=0):
        X = self.df.drop(columns='likes')
        y = self.df['likes']
        return train_test_split(X, y, test_size=test_size, random_state=random_state)