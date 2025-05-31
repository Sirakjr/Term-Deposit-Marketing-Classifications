# Term-Deposit-Marketing-Classifications

This project implements a machine learning solution for predicting term deposit subscriptions in a bank marketing campaign. The project is structured in a modular way with separate components for data preprocessing, feature engineering, and model training.

## Project Structure

```
├── notebooks/             # Jupyter notebooks
│   ├── Term_deposit_marketing.ipynb
│   └── Term_Deposit_Marketing_Modeling.ipynb
└── src/
    ├── data/
    │   └── preprocessing.py    # Data preprocessing functions
    ├── features/
    │   └── features.py        # Feature selection and engineering
    └── models/
        ├── train.py           # Model training and evaluation
        └── predict.py         # Prediction functions
```

## Features

- Modular code structure for better maintainability
- Data preprocessing pipeline
- Feature selection and importance analysis
- Random Forest model with hyperparameter tuning
- Model evaluation metrics
- Prediction functionality

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib

## Usage

The code can be used through the provided Jupyter notebooks or imported as modules in Python scripts. 