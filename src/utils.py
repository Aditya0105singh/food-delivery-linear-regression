import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_model_comparison(results):
    """
    Plot comparison of different models.
    
    Args:
        results (dict): Dictionary of model results
    """
    models = list(results.keys())
    r2_scores = [results[model]['r2'] for model in models]
    rmse_scores = [results[model]['rmse'] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² scores
    ax1.bar(models, r2_scores, color=['blue', 'green', 'orange'])
    ax1.set_title('Model Comparison - R² Scores')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1)
    
    # RMSE scores
    ax2.bar(models, rmse_scores, color=['blue', 'green', 'orange'])
    ax2.set_title('Model Comparison - RMSE Scores')
    ax2.set_ylabel('RMSE (minutes)')
    
    plt.tight_layout()
    plt.show()

def save_results(results, filename='model_results.txt'):
    """
    Save model results to a text file.
    
    Args:
        results (dict): Dictionary of model results
        filename (str): Output filename
    """
    with open(filename, 'w') as f:
        f.write("Food Delivery Time Prediction - Model Results\n")
        f.write("=" * 50 + "\n\n")
        
        for model, metrics in results.items():
            f.write(f"{model.capitalize()} Regression:\n")
            f.write(f"  R² Score: {metrics['r2']:.4f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"  MAE: {metrics['mae']:.2f}\n\n")
    
    print(f"Results saved to {filename}")
