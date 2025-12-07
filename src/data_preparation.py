import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def to_camel_case(s):
    """Convert string to CamelCase (PascalCase)."""
    # Split by underscore or space to handle snake_case or space separated
    parts = s.replace('_', ' ').split()
    # Capitalize first letter of each part
    return "".join(p[0].upper() + p[1:] for p in parts)

def ensure_camel_case_columns(df):
    """Ensure all column names are in CamelCase (PascalCase)."""
    new_cols = {col: to_camel_case(col) for col in df.columns}
    return df.rename(columns=new_cols)

def clean_data(df):
    """Clean the dataframe."""
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert TotalCharges to numeric, coercing errors to NaN
    #There are 11 instances where TotalCharges is ' ' instead of a number.Their tenure is 0.
    #That means they are new customers and have not made any payments yet.We may drop them.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing values in TotalCharges.
    df = df.dropna(subset=['TotalCharges'])
    
    # Drop customerID as it is not needed for modelling
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # Ensure all columns are in CamelCase as requested
    df = ensure_camel_case_columns(df)
    
    return df

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_dir, 'Data', 'Raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    output_path = os.path.join(base_dir, 'Data', 'Interim', 'cleaned.parquet')
    
    print(f"Loading data from {raw_data_path}...")
    try:
        df = load_data(raw_data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {raw_data_path}")
        return

    print("Cleaning data...")
    df_cleaned = clean_data(df)
    
    print("Splitting data into train and test sets (70/30)...")
    train_df, test_df = train_test_split(df_cleaned, test_size=0.3, random_state=42)
    
    train_output_path = os.path.join(base_dir, 'Data', 'Interim', 'cleaned_train.parquet')
    test_output_path = os.path.join(base_dir, 'Data', 'Interim', 'cleaned_test.parquet')
    
    print(f"Saving training data to {train_output_path}...")
    train_df.to_parquet(train_output_path, index=False)
    
    print(f"Saving test data to {test_output_path}...")
    test_df.to_parquet(test_output_path, index=False)
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
