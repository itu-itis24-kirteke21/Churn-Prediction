import streamlit as st
import requests

st.title("Telco Customer Churn Prediction")

st.write("Enter customer details to predict churn probability.")

# TODO: Create form for input
# with st.form("prediction_form"):
#     ...
#     submitted = st.form_submit_button("Predict")
#     if submitted:
#         response = requests.post("http://api:8000/predict", json=data)
#         st.write(response.json())
