FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY run_pipeline.py .
COPY run_monitoring.py .

# Create directories for artifacts and reports
RUN mkdir -p artifacts reports Data/Interim

# Expose port for FastAPI
EXPOSE 8000

# Default command runs the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
