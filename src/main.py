import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, basic_data_info
from data_cleaner import DataCleaner
from regression_models import RegressionModel
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    """
    Main pipeline for food delivery time prediction.
    """
    print("🚀 Food Delivery Time Prediction Pipeline")
    print("=" * 50)
    
    # Load data
    print("\n📊 Loading dataset...")
    df = load_dataset("data/Food_Delivery_Times.csv")
    if df is None:
        return
    
    basic_data_info(df)
    
    # Clean data
    print("\n🧹 Cleaning data...")
    cleaner = DataCleaner()
    df_clean = cleaner.handle_missing_values(df)
    df_encoded = cleaner.encode_categorical_variables(df_clean)
    df_dummies = cleaner.create_dummies(df_encoded)
    
    # Prepare features and target
    print("\n🎯 Preparing features and target...")
    X = df_dummies.drop("Delivery_Time_min", axis=1)
    y = df_dummies["Delivery_Time_min"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    print("\n🤖 Training regression models...")
    models = ['linear', 'ridge', 'lasso']
    results = {}
    
    for model_type in models:
        print(f"\nTraining {model_type} regression...")
        model = RegressionModel(model_type)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        results[model_type] = metrics
        
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['r2'])
    print(f"\n🏆 Best model: {best_model} (R² = {results[best_model]['r2']:.4f})")
    
    return results

if __name__ == "__main__":
    results = main()
