import pandas as pd
import numpy as np
import os
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import mlflow 
import mlflow.sklearn


def load_data(filepath):
    """Load data from a Parquet file."""
    return pd.read_parquet(filepath)
    


def train_model(df, params):
    """Train an XGBoost model."""
    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    mlflow.log_params(params)

    model = xgb.XGBClassifier(**params)
    model.fit(X, y)

    print("The model is being saved to MLflow in sklearn format...")
    mlflow.sklearn.log_model(model, "xgboost-model")
    
    return model, X, y

def evaluate_model(model, X, y, dataset_name="Training"):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    
    print(f"--- Model Performance on {dataset_name} Data ---")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    mlflow.log_metric(f"{dataset_name}_accuracy", acc)
    mlflow.log_metric(f"{dataset_name}_roc_auc", auc)

def save_model(model, filepath):
    """Save the trained model to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def main():
    
    mlflow.set_experiment("Telco-Churn-Test-Run")

    print("XGBoost Process Begins")

    
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, 'Data', 'Interim', 'feature_engineered_train.parquet')
    
    try:
        df = load_data(data_path)
        print("1. Data uploaded.")

    
        test_params = {
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.1,
            "objective": "binary:logistic",
            "random_state": 42
        }

        print("2. Model training and MLflow registration is starting")
        
    
        with mlflow.start_run():
            model, X, y = train_model(df, test_params)
            evaluate_model(model, X, y, dataset_name="Test-Run")
            
        print("\n PROCESS COMPLETE! Results have been saved to MLflow.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
