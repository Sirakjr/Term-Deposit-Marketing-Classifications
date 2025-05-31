import pandas as pd
import numpy as np
from typing import Tuple

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the bank marketing data.
    
    Args:
        filepath (str): Path to the raw data CSV file
        
    Returns:
        pd.DataFrame: Preprocessed dataframe ready for modeling
    """
    # Load data
    data = pd.read_csv(filepath)
    
    # Convert binary columns to 0/1
    binary_cols = ['default', 'housing', 'loan', 'y']
    for col in binary_cols:
        data[col] = (data[col] == 'yes').astype(int)
    
    # Create dummy variables for categorical columns
    cat_cols = ['job', 'marital', 'education']
    dummies_data = pd.get_dummies(data[cat_cols]).astype(int)
    
    # Combine numeric and dummy variables
    numeric_data = data.select_dtypes(include='number')
    final_data = pd.concat([numeric_data, dummies_data], axis=1)
    
    return final_data

def prepare_model_data(data: pd.DataFrame, target_col: str = 'y') -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepare data for modeling by separating features and target.
    
    Args:
        data (pd.DataFrame): Preprocessed dataframe
        target_col (str): Name of the target column
        
    Returns:
        Tuple[np.ndarray, np.ndarray, list]: X (features), y (target), and feature names
    """
    # Separate features and target
    X = data.drop(columns=[target_col]).values
    y = data[target_col].values
    feature_names = data.drop(columns=[target_col]).columns.tolist()
    
    return X, y, feature_names

def save_processed_data(data: pd.DataFrame, filepath: str) -> None:
    """
    Save processed data to CSV file.
    
    Args:
        data (pd.DataFrame): Processed dataframe to save
        filepath (str): Path where to save the processed data
    """
    data.to_csv(filepath, index=False) 