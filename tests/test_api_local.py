from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.main import app

client = TestClient(app)


def test_predict_churn():
    sample_data = {
        "SeniorCitizen": 0,
        "Tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 840.0,
        "Gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=sample_data)
        
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
        
        if response.status_code == 200:
            print("Success!")
        else:
            print("Failed!")


if __name__ == "__main__":
    test_predict_churn()
