import pandas as pd
import numpy as np
import os

def main():
    print("Generating drifted data for simulation...")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'Data', 'Interim')
    
    input_file = os.path.join(data_dir, 'cleaned_test.parquet')
    input_fe_file = os.path.join(data_dir, 'feature_engineered_test.parquet')
    
    output_file = os.path.join(data_dir, 'drift_test.parquet')
    output_fe_file = os.path.join(data_dir, 'drift_test_fe.parquet')
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return

    # 1. Drift Cleaned Data (for Monitoring)
    df = pd.read_parquet(input_file)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_drift = [c for c in num_cols if c != 'Churn']
    
    # Apply severe drift: multiply by 10 and add 5000
    df[cols_to_drift] = df[cols_to_drift] * 10 + 5000
    df.to_parquet(output_file)
    print(f"Created: {output_file}")
    
    # 2. Drift Feature Engineered Data (for Retraining)
    if os.path.exists(input_fe_file):
        df_fe = pd.read_parquet(input_fe_file)
        num_cols_fe = df_fe.select_dtypes(include=[np.number]).columns
        cols_to_drift_fe = [c for c in num_cols_fe if c != 'Churn']
        
        df_fe[cols_to_drift_fe] = df_fe[cols_to_drift_fe] * 10 + 5000
        df_fe.to_parquet(output_fe_file)
        print(f"Created: {output_fe_file}")
    
    print("\nSimulation Files Ready.")
    print("To test the pipeline with this data, run:")
    print("---------------------------------------------------------")
    print("python run_monitoring.py --current-data Data/Interim/drift_test.parquet")
    print("# If that fails (Exit Code 1), run:")
    print("python src/retrain_models.py --new-data Data/Interim/drift_test_fe.parquet")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    main()
