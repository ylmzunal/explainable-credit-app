from fastapi.testclient import TestClient
from app.main import app

#1. Create a test client instance
client = TestClient(app)

def test_predict_success():
    """
    Test that a valid feature list returns 200 OK
    and contains 'prediction' and 'probability' keys
    """
    payload = {
        "features": [
            20000, 2, 1, 0, 25, 0, 0, 0, 0, 0,
            25000, 26000, 24000, 23000, 0, 0, 0, 0,
            1000, 1500, 0, 0, 0
        ]
    }
    
    responce = client.post("/predict", json=payload)
    
    #Check HTTP status
    assert responce.status_code == 200
    
    # Check JSON response format
    json_data = responce.json()
    assert "prediction" in json_data
    assert "probability" in json_data
    
    #Check data types
    assert isinstance(json_data["prediction"], int)
    assert isinstance(json_data["probability"], float)
    