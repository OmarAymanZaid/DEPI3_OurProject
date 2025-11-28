from fastapi import FastAPI
import pandas as pd
import joblib

from healthcare_predictions.server.config import MODEL_PATH, PREPROCESSOR_PATH
from healthcare_predictions.serverTypes.server_types import Patient


# -------------------------
# Load model and preprocessor
# -------------------------

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# -------------------------
# Initialize FastAPI
# -------------------------
app = FastAPI()
   

# -------------------------
# Numeric and categorical columns
# -------------------------
numeric_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
categorical_cols = ["gender", "work_type", "ever_married", "Residence_type", "smoking_status"]

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
