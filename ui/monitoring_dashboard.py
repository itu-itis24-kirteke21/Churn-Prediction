"""
Evidently Monitoring Dashboard.

Streamlit app for viewing and generating monitoring reports.
Provides interactive visualization of data drift, quality, and model performance.
"""

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.monitor import ChurnMonitor

st.set_page_config(
    page_title="Churn Prediction Monitoring",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .status-ok { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-danger { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Churn Prediction Monitoring Dashboard")
st.markdown("Monitor data drift, quality, and model performance using Evidently AI")

# Sidebar
st.sidebar.header("Configuration")

# Data paths
default_reference = "Data/Interim/cleaned_train.parquet"
default_current = "Data/Interim/cleaned_test.parquet"

reference_path = st.sidebar.text_input("Reference Data Path", default_reference)
current_path = st.sidebar.text_input("Current Data Path", default_current)

# Evidently Cloud settings
st.sidebar.markdown("---")
st.sidebar.subheader("☁️ Evidently Cloud")
cloud_connected = os.getenv("EVIDENTLY_API_TOKEN") and os.getenv("EVIDENTLY_PROJECT_ID")

if cloud_connected:
    st.sidebar.success("✓ Connected to Evidently Cloud")
else:
    st.sidebar.warning("⚠ Not connected to Cloud")
    st.sidebar.caption("Set EVIDENTLY_API_TOKEN and EVIDENTLY_PROJECT_ID in .env")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Reference Data")
    if os.path.exists(reference_path):
        try:
            ref_data = pd.read_parquet(reference_path)
            st.success(f"✓ Loaded: {ref_data.shape[0]} rows, {ref_data.shape[1]} columns")
            with st.expander("Preview Reference Data"):
                st.dataframe(ref_data.head(10))
        except Exception as e:
            st.error(f"Error loading: {e}")
            ref_data = None
    else:
        st.warning(f"File not found: {reference_path}")
        ref_data = None

with col2:
    st.subheader("📁 Current Data")
    if os.path.exists(current_path):
        try:
            curr_data = pd.read_parquet(current_path)
            st.success(f"✓ Loaded: {curr_data.shape[0]} rows, {curr_data.shape[1]} columns")
            with st.expander("Preview Current Data"):
                st.dataframe(curr_data.head(10))
        except Exception as e:
            st.error(f"Error loading: {e}")
            curr_data = None
    else:
        st.warning(f"File not found: {current_path}")
        curr_data = None

st.markdown("---")

# Generate Reports Section
st.header("🔄 Generate Reports")

if ref_data is not None and curr_data is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        run_drift = st.button("📈 Data Drift Report", use_container_width=True)
    with col2:
        run_quality = st.button("✅ Data Quality Report", use_container_width=True)
    with col3:
        run_tests = st.button("🧪 Run Test Suite", use_container_width=True)
    with col4:
        run_all = st.button("🚀 Run All", type="primary", use_container_width=True)
    
    monitor = ChurnMonitor()
    output_dir = "reports"
    
    if run_drift or run_all:
        with st.spinner("Generating Data Drift Report..."):
            report = monitor.generate_data_drift_report(
                ref_data, curr_data,
                output_path=f"{output_dir}/data_drift_report.html"
            )
            st.success("✓ Data Drift Report generated!")
    
    if run_quality or run_all:
        with st.spinner("Generating Data Quality Report..."):
            report = monitor.generate_data_quality_report(
                ref_data, curr_data,
                output_path=f"{output_dir}/data_quality_report.html"
            )
            st.success("✓ Data Quality Report generated!")
    
    if run_tests or run_all:
        with st.spinner("Running Test Suite..."):
            results = monitor.run_data_tests(
                ref_data, curr_data,
                output_path=f"{output_dir}/test_suite_report.html"
            )
            st.success("✓ Test Suite completed!")
else:
    st.warning("⚠ Load both reference and current data to generate reports")

st.markdown("---")

# View Existing Reports
st.header("📄 View Reports")

reports_dir = Path("reports")
if reports_dir.exists():
    reports = list(reports_dir.glob("*.html"))
    if reports:
        selected_report = st.selectbox(
            "Select a report to view:",
            options=reports,
            format_func=lambda x: f"{x.stem} ({datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
        )
        
        if selected_report:
            with open(selected_report, "r") as f:
                report_html = f.read()
            
            st.components.v1.html(report_html, height=800, scrolling=True)
    else:
        st.info("No reports found. Generate reports using the buttons above.")
else:
    st.info("Reports directory not found. Generate reports to create it.")

# Footer
st.markdown("---")
st.caption("Powered by Evidently AI | Churn Prediction Monitoring")
