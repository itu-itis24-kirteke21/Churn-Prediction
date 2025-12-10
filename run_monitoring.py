#!/usr/bin/env python3
"""
Run Evidently Monitoring for Churn Prediction.

This script loads reference (training) and current (production) data,
runs all monitoring reports, and uploads results to Evidently Cloud.

Usage:
    python run_monitoring.py
    python run_monitoring.py --current-data path/to/new_data.parquet
"""

import argparse
import os
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from src.monitoring.monitor import ChurnMonitor


def main():
    parser = argparse.ArgumentParser(description='Run Evidently Monitoring')
    parser.add_argument(
        '--reference-data', 
        type=str, 
        default='Data/Interim/cleaned_train.parquet',
        help='Path to reference (training) data'
    )
    parser.add_argument(
        '--current-data', 
        type=str, 
        default='Data/Interim/cleaned_test.parquet',
        help='Path to current (production) data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Directory to save HTML reports'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Evidently Monitoring for Churn Prediction")
    print("="*60)
    
    # Check for Evidently Cloud credentials
    api_token = os.getenv("EVIDENTLY_API_TOKEN")
    project_id = os.getenv("EVIDENTLY_PROJECT_ID")
    
    if api_token and project_id:
        print("✓ Evidently Cloud credentials found")
    else:
        print("⚠ Evidently Cloud credentials not found.")
        print("  Reports will be saved locally only.")
        print("  Set EVIDENTLY_API_TOKEN and EVIDENTLY_PROJECT_ID in .env for cloud sync.")
    
    print()
    
    # Load reference data
    print(f"Loading reference data from: {args.reference_data}")
    if not os.path.exists(args.reference_data):
        print(f"Error: Reference data file not found: {args.reference_data}")
        print("Run data preparation first: python src/data_preparation.py")
        sys.exit(1)
    
    reference_data = pd.read_parquet(args.reference_data)
    print(f"  Shape: {reference_data.shape}")
    
    # Load current data
    print(f"\nLoading current data from: {args.current_data}")
    if not os.path.exists(args.current_data):
        print(f"Error: Current data file not found: {args.current_data}")
        sys.exit(1)
    
    current_data = pd.read_parquet(args.current_data)
    print(f"  Shape: {current_data.shape}")
    
    # Initialize monitor
    monitor = ChurnMonitor(api_token=api_token, project_id=project_id)
    
    # Run full monitoring
    results = monitor.run_full_monitoring(
        reference_data=reference_data,
        current_data=current_data,
        output_dir=args.output_dir
    )
    
    # Summary
    print("\n" + "="*60)
    print("MONITORING SUMMARY")
    print("="*60)
    print(f"Reports generated: {len([r for r in results.values() if r is not None])}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    
    if api_token and project_id:
        print(f"\nView dashboard: https://app.evidently.cloud")
    else:
        print(f"\nOpen reports in browser:")
        for name in ["data_drift_report.html", "data_quality_report.html", "test_suite_report.html"]:
            print(f"  - {args.output_dir}/{name}")


if __name__ == "__main__":
    main()
