import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import mlflow        
import mlflow.sklearn
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Force set the URI to ensure it goes to the DB (Absolute Path)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
db_path = os.path.join(base_dir, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")


def load_data(filepath):
    """Load data from a Parquet file."""

    return pd.read_parquet(filepath)

def train_model(df, params=None):
    """Train a Logistic Regression model."""
    if params is None:
        params = {'solver': 'liblinear', 'random_state': 42}
        
    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    mlflow.log_params(params)
    
    # Initialize and train the model
    model = LogisticRegression(**params)
    model.fit(X, y)

    print("The model is being saved to MLflow in sklearn format.")
    mlflow.sklearn.log_model(model, "logreg-model")
    
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

    # Define paths
    base_dir = os.getcwd()
    train_data_path = os.path.join(base_dir, 'Data', 'Interim', 'feature_engineered_train.parquet')
    model_output_path = os.path.join(base_dir, 'artifacts', 'logistic_regression.pkl')
    print(f"Loading training data from {train_data_path}...")
    try:
        df_train = load_data(train_data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {train_data_path}")
        return
    

    print("Training Logistic Regression model...")
    with mlflow.start_run():
        model, X_train, y_train = train_model(df_train)
        
        evaluate_model(model, X_train, y_train)
        
        print(f"Saving model to {model_output_path}...")
        save_model(model, model_output_path)
        
    print("\nPROCESS COMPLETE! Results have been saved to MLflow.")

if __name__ == "__main__":
    main()