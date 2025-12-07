import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def load_data(filepath):
    """Load data from a Parquet file."""
    return pd.read_parquet(filepath)

def train_model(df):
    """Train a Logistic Regression model."""
    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Initialize and train the model
    # Using 'liblinear' solver as it's good for smaller datasets and binary classification
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y)
    
    return model, X, y

def evaluate_model(model, X, y):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    print("--- Model Performance on Training Data ---")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

def save_model(model, filepath):
    """Save the trained model to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_data_path = os.path.join(base_dir, 'Data', 'Interim', 'feature_engineered_train.parquet')
    model_output_path = os.path.join(base_dir, 'artifacts', 'logistic_regression.pkl')
    
    print(f"Loading training data from {train_data_path}...")
    try:
        df_train = load_data(train_data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {train_data_path}")
        return

    print("Training Logistic Regression model...")
    model, X_train, y_train = train_model(df_train)
    
    evaluate_model(model, X_train, y_train)
    
    print(f"Saving model to {model_output_path}...")
    save_model(model, model_output_path)

if __name__ == "__main__":
    main()
