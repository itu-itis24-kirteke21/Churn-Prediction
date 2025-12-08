import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """Load data from a Parquet file."""
    return pd.read_parquet(filepath)

def feature_engineering(df, config=None):
    """Apply feature engineering steps based on configuration."""
    df = df.copy()
    
    # Default to one_hot if config is not provided
    encoding_strategy = 'one_hot'
    if config and 'encoding' in config:
        encoding_strategy = config['encoding']
    
    # Binary encoding
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            
    # Gender mapping
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})

    cat_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaymentMethod'
    ]
    
    # Ensure they exist
    cat_cols = [c for c in cat_cols if c in df.columns]

    if encoding_strategy == 'one_hot':
        # Get dummies
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    elif encoding_strategy == 'no_encoding':
        # Convert objects to category type for XGBoost
        for col in cat_cols:
            df[col] = df[col].astype('category')
            
    return df

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_input_path = os.path.join(base_dir, 'Data', 'Interim', 'cleaned_train.parquet')
    test_input_path = os.path.join(base_dir, 'Data', 'Interim', 'cleaned_test.parquet')
    train_output_path = os.path.join(base_dir, 'Data', 'Interim', 'feature_engineered_train.parquet')
    test_output_path = os.path.join(base_dir, 'Data', 'Interim', 'feature_engineered_test.parquet')
    
    print("Loading data...")
    try:
        train_df = load_data(train_input_path)
        test_df = load_data(test_input_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Combine for consistent encoding
    train_len = len(train_df)
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    print("Performing feature engineering...")
    combined_df_fe = feature_engineering(combined_df)
    
    # Split back
    train_df_fe = combined_df_fe.iloc[:train_len].copy()
    test_df_fe = combined_df_fe.iloc[train_len:].copy()
    
    print(f"Saving feature engineered train data to {train_output_path}...")
    train_df_fe.to_parquet(train_output_path, index=False)
    
    print(f"Saving feature engineered test data to {test_output_path}...")
    test_df_fe.to_parquet(test_output_path, index=False)
    
    print("Feature engineering complete.")
    print(f"Train shape: {train_df_fe.shape}")
    print(f"Test shape: {test_df_fe.shape}")

if __name__ == "__main__":
    main()
