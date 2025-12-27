import argparse
import pandas as pd
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Retrain models with new data.")
    parser.add_argument("--new-data", type=str, required=True, help="Path to the new data file that triggered retraining")
    args = parser.parse_args()
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root/src/.. -> Root
    data_dir = os.path.join(base_dir, 'Data', 'Interim')
    train_data_path = os.path.join(data_dir, 'feature_engineered_train.parquet')
    
    print("="*60)
    print("AUTOMATED RETRAINING SEQUENCE")
    print("="*60)
    
    # 1. Update Training Data
    print(f"1. Merging new data from {args.new_data} into training set...")
    
    if not os.path.exists(train_data_path):
        print(f"Error: Original training data not found at {train_data_path}")
        sys.exit(1)
        
    if not os.path.exists(args.new_data):
        print(f"Error: New data not found at {args.new_data}")
        sys.exit(1)
        
    try:
        df_train = pd.read_parquet(train_data_path)
        df_new = pd.read_parquet(args.new_data)
        
        # Ensure consistent columns
        common_cols = df_train.columns.intersection(df_new.columns)
        
        # Concatenate
        df_updated = pd.concat([df_train[common_cols], df_new[common_cols]], ignore_index=True)
        
        # Save back
        df_updated.to_parquet(train_data_path)
        print(f"   Success! Training set updated. New shape: {df_updated.shape} (was {df_train.shape})")
        
    except Exception as e:
        print(f"   Error updating data: {e}")
        sys.exit(1)
        
    # 2. Retrain Models
    print("\n2. Retraining XGBoost...")
    try:
        subprocess.run([sys.executable, os.path.join(base_dir, "src", "models", "trainXGBoost.py")], check=True)
    except subprocess.CalledProcessError:
        print("   XGBoost Training Failed!")
        sys.exit(1)
        
    print("\n3. Retraining Logistic Regression...")
    try:
        subprocess.run([sys.executable, os.path.join(base_dir, "src", "models", "train_LogReg.py")], check=True)
    except subprocess.CalledProcessError:
        print("   Logistic Regression Training Failed!")
        sys.exit(1)
        
    print("\nRetraining Complete. Models updated in 'artifacts/' folder.")

if __name__ == "__main__":
    main()
