import argparse
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

def load_data(filepath):
    """Load data from a Parquet file."""
    return pd.read_parquet(filepath)

def load_model(filepath):
    """Load a trained model from a pickle file."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate(model, X, y):
    """Evaluate the model on test data."""
    # Predict
    y_pred = model.predict(X)
    
    # Check if model has predict_proba
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]# take the probability of the positive class .
    else:
        y_prob = None

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    print(f"--- Model Performance on Test Data ---")
    print(f"Accuracy: {accuracy:.4f}")
    
    if y_prob is not None:
        auc = roc_auc_score(y, y_prob)
        print(f"ROC AUC: {auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

def main():
    parser = argparse.ArgumentParser(description="Predict and evaluate using a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model file (without extension), e.g., 'logistic_regression'")
    args = parser.parse_args()
    
    model_name = args.model
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_data_path = os.path.join(base_dir, 'Data', 'Interim', 'feature_engineered_test.parquet')
    model_path = os.path.join(base_dir, 'artifacts', f'{model_name}.pkl')
    
    print(f"Loading test data from {test_data_path}...")
    try:
        df_test = load_data(test_data_path)
    except FileNotFoundError:
        print(f"Error: Test data file not found at {test_data_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    # Prepare data
    if 'Churn' in df_test.columns:
        X_test = df_test.drop(columns=['Churn'])
        y_test = df_test['Churn']
    else:
        print("Error: 'Churn' column not found in test data.")
        return

    print(f"Evaluating model: {model_name}...")
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()
