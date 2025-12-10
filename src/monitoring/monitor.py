"""
Evidently AI Monitoring Module for Churn Prediction.

This module provides monitoring capabilities using Evidently
for data drift detection, quality monitoring, and model performance tracking.
"""

import os
import pandas as pd
from typing import Optional
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns, TestShareOfDriftedColumns
from evidently.tests import TestNumberOfMissingValues, TestShareOfMissingValues


class ChurnMonitor:
    """
    Monitor for Churn Prediction model using Evidently.
    
    Provides data drift detection, data quality monitoring,
    and classification performance tracking.
    """
    
    def __init__(self, api_token: Optional[str] = None, project_id: Optional[str] = None):
        """
        Initialize the monitor.
        
        Args:
            api_token: Evidently Cloud API token (optional).
            project_id: Evidently Cloud project ID (optional).
        """
        self.api_token = api_token or os.getenv("EVIDENTLY_API_TOKEN")
        self.project_id = project_id or os.getenv("EVIDENTLY_PROJECT_ID")
    
    def generate_data_drift_report(
        self, 
        reference_data: pd.DataFrame, 
        current_data: pd.DataFrame,
        save_html: bool = True,
        output_path: str = "reports/data_drift_report.html"
    ) -> Report:
        """Generate a data drift report comparing reference and current data."""
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        
        if save_html:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            report.save_html(output_path)
            print(f"Data drift report saved to: {output_path}")
        
        return report
    
    def generate_data_quality_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        save_html: bool = True,
        output_path: str = "reports/data_quality_report.html"
    ) -> Report:
        """Generate a data quality report."""
        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        
        if save_html:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            report.save_html(output_path)
            print(f"Data quality report saved to: {output_path}")
        
        return report
    
    def run_data_tests(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        max_drift_share: float = 0.3,
        save_html: bool = True,
        output_path: str = "reports/test_suite_report.html"
    ) -> TestSuite:
        """Run automated data quality and drift tests."""
        test_suite = TestSuite(tests=[
            TestNumberOfDriftedColumns(),
            TestShareOfDriftedColumns(lte=max_drift_share),
            TestNumberOfMissingValues(),
            TestShareOfMissingValues(lte=0.1),
        ])
        
        test_suite.run(reference_data=reference_data, current_data=current_data)
        
        if save_html:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            test_suite.save_html(output_path)
            print(f"Test suite report saved to: {output_path}")
        
        # Print summary
        results = test_suite.as_dict()
        passed = all(test.get("status") == "SUCCESS" for test in results.get("tests", []))
        print(f"Test Suite: {'PASSED ✓' if passed else 'FAILED ✗'}")
        
        return test_suite
    
    def run_full_monitoring(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_dir: str = "reports"
    ) -> dict:
        """Run all monitoring reports and tests."""
        print("\n" + "="*50)
        print("Running Full Monitoring Suite")
        print("="*50 + "\n")
        
        results = {}
        
        # Data Drift
        print("1. Generating Data Drift Report...")
        results["drift"] = self.generate_data_drift_report(
            reference_data, 
            current_data,
            output_path=f"{output_dir}/data_drift_report.html"
        )
        
        # Data Quality
        print("\n2. Generating Data Quality Report...")
        results["quality"] = self.generate_data_quality_report(
            reference_data,
            current_data,
            output_path=f"{output_dir}/data_quality_report.html"
        )
        
        # Test Suite
        print("\n3. Running Data Tests...")
        results["tests"] = self.run_data_tests(
            reference_data,
            current_data,
            output_path=f"{output_dir}/test_suite_report.html"
        )
        
        print("\n" + "="*50)
        print(f"Monitoring complete! Reports saved to: {output_dir}/")
        print("="*50)
        
        return results


# Legacy function for backwards compatibility
def generate_drift_report(reference_data, current_data, output_path="drift_report.html"):
    """Generate a data drift report using Evidently AI."""
    monitor = ChurnMonitor()
    monitor.generate_data_drift_report(
        reference_data, 
        current_data, 
        output_path=output_path
    )
