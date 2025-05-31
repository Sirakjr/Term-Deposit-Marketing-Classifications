import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from typing import Tuple, Dict, Any

def train_model(X: np.ndarray, y: np.ndarray, 
                test_size: float = 0.2, 
                random_state: int = 42) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train a Random Forest model with hyperparameter tuning.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[RandomForestClassifier, Dict[str, Any]]: Trained model and evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    # Initialize model and grid search
    rf = RandomForestClassifier(random_state=random_state)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Fit model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'best_params': grid_search.best_params_
    }
    
    return best_model, metrics

def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model to save
        filepath (str): Path where to save the model
    """
    joblib.dump(model, filepath) 