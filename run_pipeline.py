import argparse
import yaml
import os
import pandas as pd
import importlib 
import sys

# Add src to path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.feature_engineering import feature_engineering

def load_config(model_name):
    """Load configuration from yaml file."""
    config_path = os.path.join('config', f'{model_name}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(data_type='train'):
    """Load cleaned data."""
    filename = f'cleaned_{data_type}.parquet'
    data_path = os.path.join('Data', 'Interim', filename)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_parquet(data_path)

def get_training_module(model_name):
    """Map model name to training script module."""
    # Map config model name to python module
    mapping = {
        'xgboost': 'src.models.trainXGBoost',
        'logistic_regression': 'src.models.train_LogReg'
    }
    
    if model_name not in mapping:
        raise ValueError(f"Unknown model: {model_name}")
    
    return importlib.import_module(mapping[model_name])

def main():
    parser = argparse.ArgumentParser(description='Run Machine Learning Pipeline')
    parser.add_argument('--model', type=str, required=True, help='Model to train (xgboost, logistic_regression)')
    args = parser.parse_args()

    print(f"--- Running Pipeline for {args.model} ---")
    
    # 1. Load Config
    config = load_config(args.model)
    print("Configuration loaded.")

    # 2. Load Train Data
    print("Loading training data...")
    df_train = load_data('train')
    
    # 3. Feature Engineering on Train
    print("Running feature engineering on train data...")
    fe_config = config.get('feature_engineering', {})
    df_train_fe = feature_engineering(df_train, config=fe_config)
    print(f"Train FE complete. Shape: {df_train_fe.shape}")
    
    # 4. Train Model
    print("Training model...")
    model_config = config.get('model', {})
    params = model_config.get('params', {})
    
    # Dynamic import
    train_module = get_training_module(model_config['name'])
    
    model, X_train, y_train = train_module.train_model(df_train_fe, params)
    
    # 5. Evaluate on Train
    train_module.evaluate_model(model, X_train, y_train, dataset_name="Training")
    
    # 6. Save Artifacts
    artifact_path = os.path.join('artifacts', f'{model_config["name"]}.pkl')
    train_module.save_model(model, artifact_path)
    
    # 7. Predict on Test Data
    print("\n--- Test Prediction & Evaluation ---")
    try:
        df_test = load_data('test')
        print("Loading test data...")
        
        # Feature Engineering on Test
        df_test_fe = feature_engineering(df_test, config=fe_config)
        
        # Prepare X_test, y_test
        if 'Churn' in df_test_fe.columns:
            X_test = df_test_fe.drop(columns=['Churn'])
            y_test = df_test_fe['Churn']
        else:
            X_test = df_test_fe
            y_test = None
            
        # Check for feature mismatch
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        
        missing_cols = train_cols - test_cols
        extra_cols = test_cols - train_cols
        
        if missing_cols or extra_cols:
            print(f"Warning: feature mismatch between training and test data.")
            if missing_cols:
                print(f"  - Missing columns (will be filled with 0): {len(missing_cols)}")
            if extra_cols:
                print(f"  - Extra columns (will be dropped): {len(extra_cols)}")
        
        # Align columns (Crucial for OHE mismatches)
        # This ensures X_test has exactly the same columns as X_train in the same order
        # Missing columns are filled with 0, extra columns are dropped
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        # Evaluate if target exists
        if y_test is not None:
             train_module.evaluate_model(model, X_test, y_test, dataset_name="Test")
        else:
             print("Test data has no target column 'Churn'. Skipping evaluation.")
             
    except FileNotFoundError:
        print("Test data file not found. Skipping test evaluation.")

    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()
