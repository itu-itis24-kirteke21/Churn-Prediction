import argparse
import pandas as pd
import pickle
import os
import json
import sys
from sklearn.metrics import accuracy_score, roc_auc_score

def load_data(filepath):
    """Load data from a Parquet file."""
    return pd.read_parquet(filepath)

def load_model(filepath):
    """Load a trained model from a pickle file."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, X, y):
    """Calculate accuracy and ROC AUC."""
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
    except:
        auc = 0.5 # Fallback if predict_proba not available
    
    acc = accuracy_score(y, y_pred)
    return acc, auc

def main():
    parser = argparse.ArgumentParser(description="Compare Champion vs Challenger models.")
    parser.add_argument("--current-data", type=str, required=True, help="Path to new data for comparison")
    args = parser.parse_args()
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    artifacts_dir = os.path.join(base_dir, 'artifacts')
    
    xgb_path = os.path.join(artifacts_dir, 'xgboost_model.pkl')
    lr_path = os.path.join(artifacts_dir, 'logistic_regression.pkl')
    metadata_path = os.path.join(artifacts_dir, 'champion_metadata.json')
    
    # Load Data
    print(f"Loading comparison data: {args.current_data}")
    if not os.path.exists(args.current_data):
         print(f"Error: Data file not found: {args.current_data}")
         sys.exit(1)
         
    df = load_data(args.current_data)
    if 'Churn' not in df.columns:
        print("Error: 'Churn' column missing in comparison data.")
        sys.exit(1)
        
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Load Models
    models = {}
    try:
        models['XGBoost'] = load_model(xgb_path)
    except Exception as e:
        print(f"Warning: Could not load XGBoost: {e}")
        
    try:
        models['LogisticRegression'] = load_model(lr_path)
    except Exception as e:
        print(f"Warning: Could not load Logistic Regression: {e}")
        
    if not models:
        print("Error: No models found to compare.")
        sys.exit(1)
        
    # Evaluate
    results = {}
    print("\n--- Model Comparison ---")
    for name, model in models.items():
        acc, auc = evaluate_model(model, X, y)
        results[name] = {'accuracy': acc, 'roc_auc': auc}
        print(f"{name}: Accuracy={acc:.4f}, AUC={auc:.4f}")
        
    # Determine Champion (Simple logic: Best AUC)
    # If tie, prefer XGBoost (existing assumption)
    
    # Get current champion from metadata if exists, else default to XGBoost
    current_champion = "XGBoost"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                current_champion = meta.get("champion", "XGBoost")
        except:
            pass
            
    print(f"\nPrevious Champion: {current_champion}")
    
    # Find best model in current run
    best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
    best_score = results[best_model_name]['roc_auc']
    
    print(f"Best Performer on New Data: {best_model_name} (AUC: {best_score:.4f})")
    
    # Update Champion
    new_champion = best_model_name
    
    if new_champion != current_champion:
        print(f"!!! CHALLENGER WIN !!! {new_champion} takes the crown from {current_champion}.")
    else:
        print(f"Champion remains {current_champion}.")
        
    # Save Metadata
    metadata = {
        "champion": new_champion,
        "metrics": results,
        "last_updated": pd.Timestamp.now().isoformat()
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Champion metadata updated at {metadata_path}")

if __name__ == "__main__":
    main()
