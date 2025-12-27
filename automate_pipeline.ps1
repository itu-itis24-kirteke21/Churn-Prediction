
# Automate Pipeline execution
# Simulates a scheduled job (e.g., cron)

$ErrorActionPreference = "Stop"

Write-Host "=================================="
Write-Host "   Starting Automated Pipeline    "
Write-Host "=================================="

# 1. Activate Virtual Environment (Assuming Windows/PowerShell)
$VenvPath = ".\venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    Write-Host "Activating virtual environment..."
    . $VenvPath
}
else {
    Write-Host "Warning: Virtual environment not found at $VenvPath. Running with system python."
}

# 2. Check for Drift (Run Monitoring)
Write-Host "Running Monitoring..."
try {
    # We run the python script. 
    # Note: run_monitoring.py currently manages the logic and return codes.
    python run_monitoring.py --current-data "Data/Interim/cleaned_test.parquet"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Monitoring finished successfully. No severe drift detected."
    }
    else {
        Write-Host "Monitoring reported issues (Exit Code: $LASTEXITCODE). Initiating RETRAINING..."
        
        # 3. Automated Retraining
        # We pass the data that caused the drift as the "new data" to include in training
        python src/retrain_models.py --new-data "Data/Interim/feature_engineered_test.parquet"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Retraining completed successfully."
        }
        else {
            Write-Host "Retraining failed!"
            exit 1
        }
    }

    # 4. Champion vs Challenger Comparison
    Write-Host "Running Champion/Challenger Comparison..."
    python src/models/compare_models.py --current-data "Data/Interim/feature_engineered_test.parquet"

}
catch {
    Write-Host "Error during pipeline execution: $_"
}

Write-Host "=================================="
Write-Host "   Pipeline Complete              "
Write-Host "=================================="
