from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# -------------------------
# Load model and preprocessor
# -------------------------
model_path = "/home/omar/DEPI3_OurProject/healthcare_predictions/Models/Logistic_Regression_Tuned.pkl"
preprocessor_path = "/home/omar/DEPI3_OurProject/healthcare_predictions/Models/preprocessor.pkl"

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)  # This should be your fitted ColumnTransformer or Pipeline

# -------------------------
# Initialize FastAPI
# -------------------------
app = FastAPI()

# -------------------------
# Input data model
# -------------------------
class Patient(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str       # strings: "Yes" / "No"
    work_type: str          # "Private", "Self-employed", etc.
    residence_type: str     # "Urban" / "Rural"
    avg_glucose_level: float
    bmi: float
    smoking_status: str     # "never smoked", "smokes", etc.

# -------------------------
# Numeric and categorical columns
# -------------------------
numeric_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
categorical_cols = ["work_type", "ever_married", "residence_type", "smoking_status"]

# -------------------------
# Root endpoint
# -------------------------
@app.get("/")
def read_root():
    return {"message": "Stroke prediction API is running."}

# -------------------------
# Predict endpoint
# -------------------------
@app.post("/predict/")
def predict(patient: Patient):
    # Convert input to DataFrame
    df = pd.DataFrame([patient.dict()])

    # Preprocess input using the loaded preprocessor
    X_processed = preprocessor.transform(df)

    # Convert to DataFrame if needed (for consistency)
    df_model_input = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())

    # Predict
    prediction = model.predict(df_model_input)[0]
    probability = model.predict_proba(df_model_input)[0][1]

    # Debug logs (optional)
    print("Model input:\n", df_model_input)
    print("Prediction:", prediction, "Probability:", probability)

    return {"stroke_prediction": int(prediction), "stroke_probability": float(probability)}
