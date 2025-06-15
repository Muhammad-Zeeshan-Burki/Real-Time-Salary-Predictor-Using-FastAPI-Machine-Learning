from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pickle
import pandas as pd
import os
import uvicorn # For running the FastAPI app

# Create a FastAPI app instance
app = FastAPI(
    title="Salary Prediction API",
    description="API for predicting salaries based on job attributes using a pre-trained ML model."
)

# Define paths to the saved model and feature names
MODEL_DIR = 'model'
MODEL_PATH_PKL = os.path.join(MODEL_DIR, 'salary_predictor_model.pkl')
FEATURE_NAMES_PATH_PKL = os.path.join(MODEL_DIR, 'original_feature_names.pkl')

# Fallback to joblib if pkl files don't exist
MODEL_PATH_JOBLIB = os.path.join(MODEL_DIR, 'salary_predictor_model.joblib') 
FEATURE_NAMES_PATH_JOBLIB = os.path.join(MODEL_DIR, 'original_feature_names.joblib')

# Load the trained model and original feature names
# These will be loaded once when the application starts
try:
    # Try to load pkl files first
    if os.path.exists(MODEL_PATH_PKL) and os.path.exists(FEATURE_NAMES_PATH_PKL):
        with open(MODEL_PATH_PKL, 'rb') as f:
            model = pickle.load(f)
        with open(FEATURE_NAMES_PATH_PKL, 'rb') as f:
            original_feature_names = pickle.load(f)
        print("Model and feature names loaded from pkl files successfully.")
    else:
        # Fallback to joblib files
        model = joblib.load(MODEL_PATH_JOBLIB)
        original_feature_names = joblib.load(FEATURE_NAMES_PATH_JOBLIB)
        print("Model and feature names loaded from joblib files successfully.")
except FileNotFoundError as e:
    raise RuntimeError(f"Model files not found. Make sure model files exist in '{MODEL_DIR}' directory. Error: {e}")
except Exception as e:
    raise RuntimeError(f"Error loading model or feature names: {e}")

# Define the input data model using Pydantic
# This ensures that incoming requests have the correct structure and data types.
# The fields should match the features used during model training.
class SalaryPredictRequest(BaseModel):
    Job_Title: str = Field(..., description="The title of the job (e.g., 'Data Scientist', 'AI Engineer').")
    Experience: int = Field(..., ge=0, description="Years of experience for the job role.")
    Education_Level: str = Field(..., description="Highest education level (e.g., 'Bachelors', 'Masters', 'PhD').")
    Employment_Type: str = Field(..., description="Type of employment (e.g., 'Full-time', 'Part-time', 'Contract').")
    City: str = Field(..., description="City where the job is located (e.g., 'New York', 'London').")
    Programming_Language: str = Field(..., description="Primary programming language (e.g., 'Python', 'R', 'Java').")

    # Example values for documentation
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Job_Title": "Data Scientist",
                    "Experience": 5,
                    "Education_Level": "Masters",
                    "Employment_Type": "Full-time",
                    "City": "New York",
                    "Programming_Language": "Python"
                }
            ]
        }
    }

# Define the output data model
class SalaryPredictResponse(BaseModel):
    predicted_salary_usd: float = Field(..., description="The predicted salary in USD.")

# Health check endpoint
@app.get("/")
async def read_root():
    """
    Root endpoint for health check.
    """
    return {"message": "Salary Prediction API is running!"}

# Prediction endpoint
@app.post("/predict_salary", response_model=SalaryPredictResponse)
async def predict_salary(request: SalaryPredictRequest):
    """
    Predicts the salary based on the provided job details.
    """
    try:
        # Convert the Pydantic request model to a pandas DataFrame
        # Ensure the order of columns matches the original feature names used during training.
        input_data = pd.DataFrame([request.model_dump()])
        
        # Reorder columns to match the training data's feature order
        # This is crucial for correct preprocessing and prediction
        input_data = input_data[original_feature_names]

        # Make prediction using the loaded model pipeline
        prediction = model.predict(input_data)[0]

        # Return the predicted salary
        return SalaryPredictResponse(predicted_salary_usd=round(float(prediction), 2))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# To run this FastAPI application:
# 1. Save the model training script and run it to generate model files
# 2. Save this script as `api_app.py`.
# 3. Run from your terminal: `uvicorn api_app:app --host 0.0.0.0 --port 8000`

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

