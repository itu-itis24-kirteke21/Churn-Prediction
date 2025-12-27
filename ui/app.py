import streamlit as st
import requests
import json
import os
import pandas as pd
import time
import altair as alt
import streamlit.components.v1 as components

# --- Configuration ---
API_URL = os.getenv("API_URL", "http://api:8000/predict")
LOG_FILE = "monitoring_log.json"

st.set_page_config(page_title="Telco Churn System", layout="wide")

st.title("Telco Churn Prediction System")

# Tabs
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 System Dashboard"])

# --- TAB 1: PREDICTION ---
with tab1:
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.write("Enter customer details below to predict the probability of churn.")

    with st.form("churn_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            
        with col2:
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            
        with col3:
            st.subheader("Account")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        # Prepare payload
        payload = {
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method
        }
        
        with st.spinner("Predicting..."):
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status() 
                
                result = response.json()
                prediction = result.get("prediction")
                probability = result.get("probability")
                
                st.markdown("---")
                st.subheader("Prediction Result")
                
                if prediction == "Churn":
                    st.error(f"⚠️ **Prediction:** {prediction}")
                    st.markdown(f"**Probability:** {probability:.2%}")
                    st.write("This customer is likely to churn.")
                else:
                    st.success(f"✅ **Prediction:** {prediction}")
                    st.markdown(f"**Probability:** {probability:.2%}")
                    st.write("This customer is likely to stay.")
                    
            except Exception as e:
                st.error(f"Error: {e}")

# --- TAB 2: DASHBOARD ---
with tab2:
    st.header("System Health & Monitoring")
    
    # Auto-refresh logic (simplistic)
    if st.button("Refresh Monitor"):
        st.rerun()
        
    # Load Logs
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
            df_logs = pd.DataFrame(logs)
            
            if not df_logs.empty:
                # Metrics
                curr_acc = df_logs.iloc[-1]['accuracy']
                curr_champ = df_logs.iloc[-1]['champion']
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Current Batch Accuracy", f"{curr_acc:.2%}", delta=f"{curr_acc-0.60:.2f} vs Threshold")
                m2.metric("Active Champion Model", curr_champ)
                m3.metric("Batches Processed", len(df_logs))
                
                # Chart
                st.subheader("Accuracy Trend")
                chart = alt.Chart(df_logs).mark_line(point=True).encode(
                    x='batch_id',
                    y=alt.Y('accuracy', scale=alt.Scale(domain=[0, 1])),
                    tooltip=['batch_id', 'accuracy', 'champion']
                ).interactive()
                
                # Add threshold line
                rule = alt.Chart(pd.DataFrame({'y': [0.6]})).mark_rule(color='red').encode(y='y')
                
                st.altair_chart(chart + rule, use_container_width=True)
                
            else:
                st.warning("No data logs available yet.")
        except Exception as e:
            st.error(f"Error reading logs: {e}")
    else:
        st.info("Waiting for simulation to start... (monitoring_log.json not found)")
        
    st.markdown("---")
    st.subheader("Evidently Reports")
    
    # List reports
    report_dir = "reports"
    reports_map = {
        "Data Drift": "data_drift_report.html",
        "Data Quality": "data_quality_report.html",
        "Test Suite": "test_suite_report.html"
    }
    
    # Helper to read html
    def get_html(fname):
        path = os.path.join(report_dir, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    # Display selector
    selected_report = st.selectbox("Select Report to View", list(reports_map.keys()))

    if st.button("View Report"):
        fname = reports_map[selected_report]
        html_content = get_html(fname)
        
        if html_content:
             st.success(f"Displaying {selected_report}")
             components.html(html_content, height=1000, scrolling=True)
        else:
             st.error(f"Report '{fname}' not found. It may not have been generated yet.")

