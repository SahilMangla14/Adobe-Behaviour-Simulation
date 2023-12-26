from data_preprocessing import DataPreprocessor
from ann import NeuralNetwork
from sklearn.impute import SimpleImputer
import torch
import joblib
from helpers import evaluate_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# df = pd.read_csv('./data_company/df_final.csv')
# df = pd.read_csv('./data_time/df_final.csv')
df = pd.read_csv('./custom/df_final.csv')

# Instantiate the model
model = NeuralNetwork(input_size=df.shape[1]) 
model.load_state_dict(torch.load('./Results/regression_model.pth'))
scaler_likes = joblib.load('./Results/scaler_likes.joblib')
scaler_company_rank = joblib.load('./Results/scaler_company_rank.joblib')
scaler_epoch_time = joblib.load('./Results/scaler_epoch_time.joblib')
scaler_username_rank = joblib.load('./Results/scaler_username_rank.joblib')


df['company_rank'] = scaler_company_rank.transform(df['company_rank'].values.reshape(-1,1))
df['epoch_time'] = scaler_epoch_time.transform(df['epoch_time'].values.reshape(-1,1))
df['username_rank'] = scaler_username_rank.transform(df['username_rank'].values.reshape(-1,1))

# Convert data to NumPy arrays and handle NaN values
df_test_array = SimpleImputer(strategy='mean').fit_transform(df.values)

X_test_tensor = torch.Tensor(df_test_array)

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)

# print(y_pred)

# y_test_np = y_test_tensor.cpu().numpy()
y_pred_np = y_pred.cpu().numpy()

# y_test_inv = scaler_likes.inverse_transform(y_test_np)
y_pred_inv = scaler_likes.inverse_transform(y_pred_np)

df_predictions = pd.DataFrame(y_pred_inv, columns=['Predictions'])
# df_predictions.to_csv('./Predictions/time_dataset_predictions.csv', index = False)
df_predictions.to_csv('./custom/test_adobe_predictions.csv', index = False)
print("Predicted Likes : ", df_predictions.iloc[0])

# # Stack the arrays side by side
# comparison_result = np.column_stack((y_test_inv, y_pred_inv))

# # Create a DataFrame for better visualization
# comparison_df = pd.DataFrame(comparison_result, columns=['y_test', 'y_pred'])

# comparison_df.to_csv('./Results/ann_results2.csv', index=False)

