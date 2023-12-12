import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor
from ann import NeuralNetwork
from trainer import Trainer
from helpers import evaluate_model, plot_scatter, plot_residuals, analyze_residuals_levels

def main():
    data_preprocessor = DataPreprocessor('./df_final.csv')
    data_preprocessor.scale_target_variable()
    data_preprocessor.scale_features()
    data_preprocessor.drop_na_rows()

    X_train, X_test, y_train, y_test = data_preprocessor.split_data()

    # Convert data to NumPy arrays and handle NaN values
    X_train_array = SimpleImputer(strategy='mean').fit_transform(X_train.values)
    X_test_array = SimpleImputer(strategy='mean').fit_transform(X_test.values)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.Tensor(X_train_array)
    y_train_tensor = torch.Tensor(y_train.values.reshape(-1, 1))
    X_test_tensor = torch.Tensor(X_test_array)
    y_test_tensor = torch.Tensor(y_test.values.reshape(-1, 1))

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = NeuralNetwork(input_size=X_train.shape[1])
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    trainer = Trainer(model, criterion, optimizer, train_loader, device)
    num_epochs = 50
    trainer.train(num_epochs)

    comparison_df = evaluate_model(model, criterion, X_test_tensor, y_test_tensor, data_preprocessor.scaler_likes, device)

    # Example: Plot scatter
    plot_scatter(comparison_df['y_test'].values, comparison_df['y_pred'].values)

    # Example: Plot residuals
    plot_residuals(comparison_df, 'y_test', 'y_pred', levels=[1, 2])

    # Example: Analyze residuals levels
    levels_to_analyze = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5]
    counts = analyze_residuals_levels(comparison_df, 'y_test', 'y_pred', levels=levels_to_analyze)

    print("Results falling within specified levels of variation:")
    for level, count in counts.items():
        print(f"{count} results within {level} of variation.")


if __name__ == "__main__":
    main()
