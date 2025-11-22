from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_drift_report(reference_data, current_data, output_path="drift_report.html"):
    """
    Generate a data drift report using Evidently AI.
    
    Args:
        reference_data: Training data.
        current_data: New incoming data.
        output_path: Where to save the HTML report.
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(output_path)
