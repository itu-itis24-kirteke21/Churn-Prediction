
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
} else {
    Write-Host "Warning: Virtual environment not found at $VenvPath. Running with system python."
}

# 2. Check for Drift (Run Monitoring)
Write-Host "Running Monitoring..."
try {
    # We run the python script. 
    # Note: run_monitoring.py currently manages the logic and return codes.
    python run_monitoring.py --current-data "Data/Interim/cleaned_test.parquet"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Monitoring finished successfully."
    } else {
        Write-Host "Monitoring reported issues (Exit Code: $LASTEXITCODE)"
    }
} catch {
    Write-Host "Error running monitoring script: $_"
}

Write-Host "=================================="
Write-Host "   Pipeline Complete              "
Write-Host "=================================="
