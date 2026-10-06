# ML-in-Production-workshop

## Run the API

From this directory, install the dependencies and start FastAPI:

```powershell
pip install -r requirements.txt
uvicorn main:app --reload
```

The model is saved with `joblib` and the project uses scikit-learn `1.5.2`.
Keep the scikit-learn version consistent when creating or replacing `model.pkl`.

Open the interactive API documentation at http://127.0.0.1:8000/docs.

Check that the model loaded correctly at http://127.0.0.1:8000/health.

### Example request

Send a `POST` request to `/predict` with the same feature values used during training:

```json
{
  "airline": "AirAsia",
  "source": "Bangalore",
  "departure": "Afternoon",
  "arrival": "Morning",
  "destination": "Kolkata",
  "flight": "AI-676",
  "class": "Business",
  "stops": "zero",
  "duration": 12.0,
  "days_left": 26
}
```

The response is returned as:

```json
{"predicted_price": 12345.67}
```
