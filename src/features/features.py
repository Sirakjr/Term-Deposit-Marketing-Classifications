import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from typing import List, Tuple

def select_features(X: np.ndarray, y: np.ndarray, feature_names: List[str], k: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Select top k features using f_classif (ANOVA F-value).
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        feature_names (List[str]): List of feature names
        k (int): Number of top features to select
        
    Returns:
        Tuple[np.ndarray, List[str]]: Selected features and their names
    """
    # Initialize selector
    selector = SelectKBest(score_func=f_classif, k=k)
    
    # Fit and transform
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    return X_selected, selected_features

def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (List[str]): List of feature names
        
    Returns:
        pd.DataFrame: DataFrame with feature names and their importance scores
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    return importance.sort_values('importance', ascending=False) 