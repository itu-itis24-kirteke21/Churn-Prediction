import pickle
import pandas as pd
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

app = FastAPI(title="Telco Churn Prediction API")

# --- Configuration & Model Loading ---
MODEL_PATH = "artifacts/xgboost.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Check if running in Docker or local
        path = MODEL_PATH
        if not os.path.exists(path):
            # Try finding it relative to this file if running locally not from root
            # This file is in src/api/main.py, artifacts is in artifacts/
            # So ../../artifacts/xgboost.pkl
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(base_dir, "artifacts", "xgboost.pkl")
            
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # We don't raise here to allow API to start, but predict will fail
        
# --- Data Schema ---
class CustomerData(BaseModel):
    # Numeric features
    SeniorCitizen: int
    Tenure: int
    MonthlyCharges: float
    TotalCharges: float
    
    # Binary/Categorical features (Strings)
    Gender: Literal["Male", "Female"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: str

# --- Preprocessing ---
def preprocess_input(data: CustomerData, model_features: list) -> pd.DataFrame:
    # Convert Pydantic model to dict
    input_dict = data.dict()
    
    # Rename keys if necessary to match model features (none needed based on inspection)
    # The model expects "Gender", "SeniorCitizen", "Partner", ...
    
    df = pd.DataFrame([input_dict])
    
    # Binary Encodings
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        
    # Gender Encoding
    df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
    
    # Ensure categorical columns are treated as category dtype if model expects it
    # Based on feature_engineering.py, if 'no_encoding' was used (which implies XGBoost native Cat support),
    # we cast to 'category'.
    # List of categoricals from feature_engineering.py
    cat_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaymentMethod'
    ]
    
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    # Reorder columns to match model training
    # Add missing columns if any (shouldn't be with strict schema, but for safety)
    for feature in model_features:
        if feature not in df.columns:
            # Check if it's one of the columns we might have missed or transformed differently
            pass
            
    df = df[model_features]
    
    return df

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get feature names from model
        if hasattr(model, "feature_names_in_"):
            feature_names = model.feature_names_in_
        else:
            # Fallback based on known list if attribute missing (older sklearn/xgboost?)
            # But we saw it has it in our inspection.
             raise HTTPException(status_code=500, detail="Model does not have feature_names_in_")

        # Preprocess
        X = preprocess_input(data, feature_names)
        
        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        result = "Churn" if prediction == 1 else "Not Churn"
        
        return {
            "prediction": result,
            "probability": float(probability),
            "churn_value": int(prediction)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
