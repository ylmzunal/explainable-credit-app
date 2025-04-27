from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_success():
    """
    Tests that a valid feature list returns 200 OK
    and contains 'prediction' and 'probability' keys.
    """
    payload = {
        "features": [
            20000, 2, 1, 0, 25, 0, 0, 0, 0, 0,
            25000, 26000, 24000, 23000, 0, 0, 0, 0,
            1000, 1500, 0, 0, 0
        ]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert "probability" in json_data
    assert isinstance(json_data["prediction"], int)
    assert isinstance(json_data["probability"], float)

def test_predict_invalid_input():
    """
    Tests that sending invalid JSON returns a 422 Unprocessable Entity error.
    """
    bad_payload = {
        "wrong_key": [1, 2, 3]  # wrong field name, not 'features'
    }

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422  # Expect validation error