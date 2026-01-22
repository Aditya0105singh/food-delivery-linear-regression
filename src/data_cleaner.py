import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataCleaner:
    """
    Handle data cleaning and preprocessing for food delivery dataset.
    """
    
    def __init__(self):
        self.label_encoders = {}
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Handle categorical missing values with mode
        categorical_cols = ['Weather', 'Traffic_Level', 'Time_of_Day']
        for col in categorical_cols:
            if col in df_clean.columns:
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_value)
        
        # Handle numerical missing values with mean
        numerical_cols = ['Courier_Experience_yrs']
        for col in numerical_cols:
            if col in df_clean.columns:
                mean_value = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(mean_value)
        
        print("Missing values handled successfully")
        return df_clean
    
    def encode_categorical_variables(self, df):
        """
        Encode categorical variables using label encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Encoded dataframe
        """
        df_encoded = df.copy()
        categorical_columns = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
        
        print("Categorical variables encoded successfully")
        return df_encoded
    
    def create_dummies(self, df):
        """
        Create dummy variables for categorical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with dummy variables
        """
        cols_to_dummy = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
        df_dummies = pd.get_dummies(df, columns=cols_to_dummy)
        df_dummies = df_dummies.astype(int)
        
        print(f"Dummy variables created. New shape: {df_dummies.shape}")
        return df_dummies
