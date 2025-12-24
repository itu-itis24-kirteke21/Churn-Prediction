
import pandas as pd
import numpy as np
import os

def create_drifted_data(output_path="Data/Interim/drifted_test.parquet"):
    """
    Creates a synthetic dataset designed to trigger data drift alerts.
    """
    print("Generating drifted data...")
    
    # Load original schematic (or just create synthetic data from scratch)
    # We'll create a dataframe with the same columns but wild values
    
    n_samples = 1000
    
    data = {
        'Gender': np.random.choice(['Alien', 'Predator'], n_samples), # New categories -> Drift
        'SeniorCitizen': np.zeros(n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'Tenure': np.random.randint(100, 200, n_samples), # Out of usual range -> Drift
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'MonthlyCharges': np.random.uniform(500, 1000, n_samples), # Massive drift
        'TotalCharges': np.random.uniform(50000, 100000, n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"Drifted data saved to: {output_path}")

if __name__ == "__main__":
    create_drifted_data()
