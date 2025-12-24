# Telco Customer Churn Prediction System

## Project Overview
This project is an end-to-end Machine Learning Operations (MLOps) system designed to predict customer churn in the telecommunications industry. It encompasses the entire lifecycle from data preparation and model training to deployment and monitoring.

The system is built to be modular, containerized, and easily deployable, featuring:
- **FastAPI** for model serving.
- **Streamlit** for a user-friendly frontend interface.
- **XGBoost** as the core predictive model.
- **Docker & Docker Compose** for container orchestration.
- **MLflow** for experiment tracking (integrated in pipeline).

## Team Members
1. Furkan Kırteke
2. Burak Koçoğlu
3. Defne Yıldırım

## Quick Start (Docker)
The easiest way to run the application is using Docker Compose. This ensures all dependencies and services are set up correctly.

### Prerequisites
- Docker Desktop installed and running.

### Running the Application
1. **Clone/Open the repository**.
2. **Build and Start Services**:
   ```bash
   docker-compose up -d --build
   ```
   This command builds the API and UI images and starts them along with MLflow.

### Accessing the Services
| Service | URL | Description |
|---------|-----|-------------|
| **Streamlit UI** | [http://localhost:8501](http://localhost:8501) | User interface for entering customer data and getting predictions. |
| **API Documentation** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive Swagger UI for the FastAPI backend. |
| **MLflow UI** | [http://localhost:5000](http://localhost:5000) | Dashboard for tracking model experiments. |

---

## Detailed Directory Structure

### `src/` - Source Code
Core logic of the project.

- **`src/api/`**
  - `main.py`: The FastAPI application entry point. Handles model loading, input validation (Pydantic), preprocessing, and serving predictions.

- **`src/models/`**
  - `trainXGBoost.py`: Script to train the XGBoost model.
    - **Usage**: `python src/models/trainXGBoost.py` (usually run via pipeline).
    - **Output**: Saves model to `artifacts/xgboost.pkl`.
  - `predict.py`: Standalone script for batch prediction on parquet files.
    - **Arguments**: `--model <model_name>` (e.g., `xgboost`).
  - `train_LogReg.py`: Logistic Regression training script (alternative baseline).

- **`src/data_preparation.py`**
  - Cleans raw data, handles missing values, and normalizes formats.

- **`src/feature_engineering.py`**
  - Transforms cleaned data into model-ready features (encoding categorical variables, scaling, etc.).
  - Handles **One-Hot Encoding** or **Categorical** types based on config.

- **`src/monitoring/`**
  - Contains scripts for Evidently AI to detect data drift (if applicable).

### `ui/` - User Interface
- `app.py`: The Streamlit application. It renders a form for user input, communicates with the API, and displays results.

### `config/` - Configuration
- `xgboost.yaml`: Hyperparameters and settings for XGBoost training (e.g., `max_depth`, `learning_rate`, `enable_categorical`).
- `logistic_regression.yaml`: Settings for Logistic Regression.

### `artifacts/` - Model Storage
- Stores trained model binaries (e.g., `xgboost.pkl`). This directory is mounted into the API container.

### Root Files
- **`docker-compose.yml`**: Defines the multi-container application (API, UI, MLflow).
- **`Dockerfile`**: Instructions for building the API image (Python 3.11, XGBoost 3.1.2).
- **`requirements.txt`**: Python dependencies.
- **`run_pipeline.py`**: A master script to orchestrate the data prep -> feature engineering -> training workflow locally.

---

## API Documentation
The API exposes a single main endpoint for prediction.

### Endpoint: `POST /predict`
**Request Body (JSON):**
```json
{
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
```

**Response (JSON):**
```json
{
  "prediction": "Not Churn",
  "probability": 0.1234,
  "churn_value": 0
}
```

---

## Local Development (Without Docker)
If you wish to run scripts manually or develop locally:

1. **Create Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Pipeline (Prep > Feature > Train)**:
   ```bash
   python run_pipeline.py
   ```
   *Note: Ensure raw data is in `Data/Raw/`.*

4. **Run API Locally**:
   ```bash
   uvicorn src.api.main:app --reload
   ```

5. **Run UI Locally**:
   ```bash
   streamlit run ui/app.py
   ```
