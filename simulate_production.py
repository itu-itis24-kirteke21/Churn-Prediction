import time
import requests
import pandas as pd
import numpy as np
import os
import json
import subprocess
import shutil

API_URL = "http://localhost:8000/predict"
RELOAD_URL = "http://localhost:8000/reload"
LOG_FILE = "monitoring_log.json"
DATA_PATH = "Data/Interim/cleaned_test_master.parquet"
DRIFT_FE_PATH = "Data/Interim/drift_test_fe.parquet" # Pre-created for simplicity or created on fly

def log_metrics(batch_id, accuracy, champion):
    """Log metrics to a JSON file for the dashboard."""
    entry = {
        "batch_id": batch_id,
        "accuracy": accuracy,
        "champion": champion,
        "timestamp": time.time()
    }
    
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        except:
            pass
            
    logs.append(entry)
    # Keep last 100
    logs = logs[-100:]
    
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f)

def get_current_champion():
    try:
        with open("artifacts/champion_metadata.json", "r") as f:
            return json.load(f).get("champion", "Unknown")
    except:
        return "Unknown"

def main():
    print("Starting Production Simulation...")

    # Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}")
        return
    
    # We will iterate through the test set in batches
    full_df = pd.read_parquet(DATA_PATH)
    batch_size = 50
    
    # Create drift trigger data if not exists (using helper logic inline)
    # We need a file to pass to the pipeline when drift happens
    # (Assuming simulate_drift.py was run or we do it here)
    
    for i in range(0, len(full_df), batch_size):
        batch = full_df.iloc[i:i+batch_size].copy()
        
        # --- DRIFT INDUCTION ---
        # Slowly increase drift factor
        # e.g. from batch 10 onwards, start drifting
        drift_factor = 0
        if i > 200: 
            drift_factor = (i - 200) * 10 
            
        if drift_factor > 0:
            num_cols = batch.select_dtypes(include=[np.number]).columns
            cols_to_drift = [c for c in num_cols if c != 'Churn']
            batch[cols_to_drift] = batch[cols_to_drift] + drift_factor
            print(f"Batch {i//batch_size}: Applying Drift Factor +{drift_factor}")
        
        # --- PREDICTION ---
        correct = 0
        for _, row in batch.iterrows():
            # Prepare payload
            payload = row.drop('Churn').to_dict()
            # Fix types for JSON serialization
            for k, v in payload.items():
                if isinstance(v, (np.integer, np.int64)):
                    payload[k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    payload[k] = float(v)
            
            try:
                resp = requests.post(API_URL, json=payload)
                if resp.status_code == 200:
                    pred = resp.json()['churn_value']
                    
                    # Handle Churn format (String 'Yes'/'No' vs Int 1/0)
                    actual = row['Churn']
                    if isinstance(actual, str):
                        actual = 1 if actual == 'Yes' else 0
                        

                         
                    if pred == actual:
                        correct += 1
                else:
                    print(f"API Error: {resp.status_code} - {resp.text}")
            except Exception as e:
                print(f"Connection Error: {e}")
                
        accuracy = correct / len(batch)
        champion = get_current_champion()
        
        print(f"Batch {i//batch_size} | Accuracy: {accuracy:.2f} | Champion: {champion}")
        
        log_metrics(i//batch_size, accuracy, champion)
        
        # --- TRIGGER RETRAINING ---
        if accuracy < 0.60:
            print("!!! ACCURACY DROP DETECTED (< 0.60) !!!")
            print("Initiating Pulse Correction (Pipeline)...")
            
            # 1. Create a specific drifted file for the pipeline to consume
            # This 'mock' file represents the 'collected' data that is drifting
            drift_file = "Data/Interim/production_stream.parquet"
            batch.to_parquet(drift_file)
            
            # We also need a feature engineered version for the training script
            # For this simulation, we'll cheat slightly and use the previously generated fe file
            # OR typically we'd run the FE pipeline step here. 
            # We will assume 'Data/Interim/drift_test_fe.parquet' exists from previous steps 
            # or we create a simple copy for the demo to succeed.
            
            fe_target = "Data/Interim/feature_engineered_test.parquet" # The pipeline uses this input by default in the script we wrote
            # To make the 'compare_models' work with this new tough data, we should overwrite the test set it uses?
            # Or better, we pass arguments. 
            # In automate_pipeline.ps1, we hardcoded inputs. 
            # To make this seamless without changing ps1 too much:
            # We will rely on the fact that 'retrain_models.py' takes an argument.
            
            # Let's run the pipeline!
            # We skip monitoring step call here and go straight to "Force Retrain" logic or 
            # run the full script. 
            # Running full script is safer as it simulates the real cron job.
            
            # BUT: The pipeline reads "Data/Interim/cleaned_test.parquet". 
            # We should overwrite that file with our drifted batch to force the pipeline to fail and retrain!
            shutil.copy(drift_file, "Data/Interim/cleaned_test.parquet")
            
            # We also need the FE version for the retraining part.
            # Ideally we run feature_engineering.py. 
            # For demo speed: We assume the pipeline deals with it or we just trigger the retrain script directly.
            
            # Let's TRY running the pipeline.
            try:
                subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", "automate_pipeline.ps1"], check=False)
                
                # Reload API
                requests.post(RELOAD_URL)
                print("System Reloaded.")
                
            except Exception as e:
                print(f"Pipeline execution failed: {e}")
                
            # Reset drift to avoid infinite loop in this demo? 
            # or let it recover. Ideally it recovers because model is retrained on this data.
            
        time.sleep(1) # Simulate time gap

if __name__ == "__main__":
    main()
