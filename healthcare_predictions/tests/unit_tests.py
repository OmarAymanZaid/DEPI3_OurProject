from healthcare_predictions.server.server import predict, app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_predict_endpoint():
    # Sample valid input matching Patient model
    sample_patient = {
        "gender": "Male",
        "age": 45.0,
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 120.5,
        "bmi": 28.7,
        "smoking_status": "never smoked"
    }

    # Make the POST request
    response = client.post("/predict/", json=sample_patient)

    # Check the request was successful
    assert response.status_code == 200

    data = response.json()

    # The endpoint must return these keys
    assert "stroke_prediction" in data
    assert "stroke_probability" in data

    # Ensure they are the correct types
    assert isinstance(data["stroke_prediction"], int)
    assert isinstance(data["stroke_probability"], float)
