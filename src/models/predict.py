import numpy as np
import pandas as pd
import joblib
from typing import Union, List

def load_model(filepath: str):
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        The loaded model
    """
    return joblib.load(filepath)

def predict(model, X: np.ndarray) -> np.ndarray:
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        X (np.ndarray): Feature matrix
        
    Returns:
        np.ndarray: Predicted class labels
    """
    return model.predict(X)

def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """
    Get probability estimates for each class.
    
    Args:
        model: Trained model
        X (np.ndarray): Feature matrix
        
    Returns:
        np.ndarray: Probability estimates for each class
    """
    return model.predict_proba(X)

def format_predictions(predictions: np.ndarray, 
                      probabilities: np.ndarray,
                      customer_ids: Union[List, np.ndarray] = None) -> pd.DataFrame:
    """
    Format predictions and probabilities into a readable DataFrame.
    
    Args:
        predictions (np.ndarray): Predicted class labels
        probabilities (np.ndarray): Probability estimates
        customer_ids (Union[List, np.ndarray], optional): Customer IDs
        
    Returns:
        pd.DataFrame: Formatted predictions with probabilities
    """
    results = pd.DataFrame({
        'predicted_class': predictions,
        'probability_0': probabilities[:, 0],
        'probability_1': probabilities[:, 1]
    })
    
    if customer_ids is not None:
        results['customer_id'] = customer_ids
        
    return results 