import requests
import json

def test_predict_churn_docker():
    url = "http://localhost:8000/predict"
    
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

    try:
        response = requests.post(url, json=sample_data)
        
        print(f"Status Code: {response.status_code}")
        try:
            print("Response JSON:", json.dumps(response.json(), indent=2))
        except:
            print("Response Text:", response.text)
            
        if response.status_code == 200:
            print("Success! The Docker API is responding correctly.")
        else:
            print("Failed! The API returned an error.")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API at http://localhost:8000.")
        print("Make sure the Docker container is running and port 8000 is exposed.")

if __name__ == "__main__":
    test_predict_churn_docker()
