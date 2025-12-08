import pandas as pd
import numpy as np
import os
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def load_data(filepath):
    """Load data from a Parquet file."""
    return pd.read_parquet(filepath)

def train_model(df, params):
    """Train an XGBoost model."""
    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Initialize and train the model
    # Enable categorical support if using categoricals
    if params.get('enable_categorical', False):
        model = xgb.XGBClassifier(**params)
    else:
        # If not explicit, rely on default or other params
        # But 'enable_categorical' is a nice flag we added to our config
        # Clean params to remove our custom flag if XGBoost doesn't want it (though enable_categorical is valid in recent versions)
        # Actually XGBoost >= 1.5 supports enable_categorical=True in constructor
        model = xgb.XGBClassifier(**params)

    model.fit(X, y)
    
    return model, X, y

def evaluate_model(model, X, y, dataset_name="Training"):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    print(f"--- Model Performance on {dataset_name} Data ---")
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

if __name__ == "__main__":
    # Standalone run not fully supported without config injection, 
    # but could be added if needed. For now designed to be called by run_pipeline.
    pass
