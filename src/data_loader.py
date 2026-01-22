import pandas as pd

def load_dataset(file_path):
    """
    Load the food delivery dataset.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def basic_data_info(df):
    """
    Display basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    if df is not None:
        print("\nDataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Missing values:\n{df.isnull().sum()}")
