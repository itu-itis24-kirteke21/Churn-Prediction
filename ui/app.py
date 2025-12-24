import streamlit as st
import requests
import json
import os

# --- Configuration ---
API_URL = os.getenv("API_URL", "http://api:8000/predict")

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.title("Telco Customer Churn Prediction")
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
            response.raise_for_status() # Raise error for bad status codes
            
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
                
            with st.expander("See Request/Response Details"):
                st.json(payload)
                st.json(result)
                
        except requests.exceptions.ConnectionError:
            st.error(f"Error: Could not connect to API at `{API_URL}`. Is the API service running?")
        except requests.exceptions.HTTPError as e:
            st.error(f"API Error: {e}")
            try:
                st.json(response.json())
            except:
                st.write(response.text)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
