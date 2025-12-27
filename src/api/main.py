import pickle
import json
import pandas as pd
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

app = FastAPI(title="Telco Churn Prediction API")

# --- Configuration & Model Loading ---
# --- Configuration & Model Loading ---
MODEL_DIR = "artifacts"
METADATA_PATH = os.path.join(MODEL_DIR, "champion_metadata.json")
model = None

def get_champion_path():
    """Determine the path of the current champion model."""
    # Default to XGBoost if no metadata
    model_file = "xgboost_model.pkl" 
    
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                meta = json.load(f)
                champion = meta.get("champion", "XGBoost")
                
            if champion == "LogisticRegression":
                model_file = "logistic_regression.pkl"
            else:
                model_file = "xgboost_model.pkl"
                
            print(f"Selecting Champion Model: {champion} ({model_file})")
        except Exception as e:
            print(f"Error reading metadata, defaulting to XGBoost: {e}")
            
    return os.path.join(MODEL_DIR, model_file)

def _load_model_logic():
    global model
    try:
        path = get_champion_path()
        if not os.path.exists(path):
            # Fallback for local run vs docker
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(base_dir, path)
            
        if not os.path.exists(path):
             print(f"Error: Model file not found at {path}")
             return

        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {path}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.on_event("startup")
def load_model():
    _load_model_logic()

@app.post("/reload")
def reload_model():
    """Endpoint to trigger model reloading."""
    _load_model_logic()
    return {"status": "Model reloaded", "champion_path": get_champion_path()}

        
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
    
    # --- FEATURE ENGINEERING (Must match src/feature_engineering.py) ---
    
    # 1. Binary Encoding
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        
    # 2. Gender Encoding
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
    
    # 3. Handle Categorical Columns (One-Hot Logic to match training default)
    # We must replicate the 'one_hot' strategy by default as typical models (LogReg, basic XGB) use it unless strictly separated.
    # If the loaded model expects raw categories (XGB special), the 'model_features' check below handles appropriate selection,
    # BUT if 'model_features' has OHE cols, we MUST generate them.
    
    # Check if we need to OHE?
    # Simple heuristic: If model_features contains "Contract_Two year", we need to OHE.
    needs_ohe = any('_' in feat for feat in model_features if feat not in df.columns)
    
    if needs_ohe:
        cat_cols = [
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaymentMethod'
        ]
        df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)
            
    # 4. ALIGNMENT (Crucial for Single-Row Prediction)
    # Ensure all model features exist, fill missing with 0
    for feature in model_features:
        if feature not in df.columns:
            df[feature] = 0
            
    # Remove extra columns that model doesn't know
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
