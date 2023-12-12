import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def evaluate_model(model, criterion, X_test_tensor, y_test_tensor, scaler_likes, device):
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    
    with torch.no_grad():
        model.eval()
        y_pred = model(X_test_tensor)
        test_loss = criterion(y_pred, y_test_tensor)
        print(f'Mean Squared Error on Test Set: {test_loss.item():.4f}')

    # Convert tensors to NumPy arrays
    y_test_np = []
    y_pred_np = []
    if(device != "cpu"):
        y_test_np = y_test_tensor.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
    else:
        y_test_np = y_test_tensor.numpy()
        y_pred_np = y_pred.numpy()
        
    # Inverse transform using the scaler
    y_test_inv = scaler_likes.inverse_transform(y_test_np)
    y_pred_inv = scaler_likes.inverse_transform(y_pred_np)

    # Stack the arrays side by side
    comparison_result = np.column_stack((y_test_inv, y_pred_inv))

    # Create a DataFrame for better visualization
    comparison_df = pd.DataFrame(comparison_result, columns=['y_test', 'y_pred'])

    # Display the DataFrame
    comparison_df.to_csv('./ann_results.csv', index=False)

    return comparison_df


def plot_scatter(y_test_np, y_pred_np):
    plt.scatter(y_test_np, y_pred_np)
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.save('./scatter_plot.png')
    # plt.show()


def plot_residuals(df, actual_col, predicted_col, levels=None):
    residuals = df[actual_col] - df[predicted_col]

    plt.figure(figsize=(8, 6))
    plt.scatter(df[predicted_col], residuals, color='green')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.axhline(y=0, color='red', linestyle='--', label='Residuals Mean')

    if levels:
        for level in levels:
            sd = np.std(residuals)
            plt.axhline(y=level * sd, color='blue', linestyle='-', label=f'{level} SD')

    plt.legend()
    plt.grid(True)
    plt.save('./residual_plot.png')
    # plt.show()


def analyze_residuals_levels(df, actual_col, predicted_col, levels=None):
    residuals = df[actual_col] - df[predicted_col]
    sd = np.std(residuals)

    results_count = {}

    if levels:
        for i, level in enumerate(levels):
            counts = 0
            count_within_level = np.sum(np.abs(residuals) <= level * sd)
            counts_prev_level = np.sum(np.abs(residuals) <= levels[max(0, i - 1)] * sd)

            if i:
                counts = count_within_level - counts_prev_level
            else:
                counts = count_within_level
            results_count[f'{level} SD'] = counts

    return results_count