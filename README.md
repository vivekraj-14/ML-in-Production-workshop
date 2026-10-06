# Flight Price Prediction API

A FastAPI application that predicts flight ticket prices using a trained
Decision Tree regression model stored in `model.pkl`. The project includes a
browser-based prediction form and JSON API endpoints.

## Project structure

```text
.
├── main.py            # FastAPI app, GUI, validation, and prediction logic
├── model.pkl          # Trained scikit-learn model
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container configuration
└── .dockerignore
```

## Run locally

Use Python 3.11 and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

The web interface is available at `/`. The API documentation is available at
`/docs`.

## API endpoints

### Health check

Request:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected result:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Prediction

Request:

```powershell
$body = @{
    airline = "AirAsia"
    source = "Bangalore"
    departure = "Afternoon"
    arrival = "Morning"
    destination = "Kolkata"
    flight = "AI-676"
    class = "Business"
    stops = "zero"
    duration = 12.0
    days_left = 26
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Example result:

```json
{
  "predicted_price": 27108.62
}
```

The exact prediction depends on the current `model.pkl` file and the supplied
feature values.

## Input fields

The API expects:

- `airline`, `source`, `destination`, `departure`, `arrival`, and `flight` to
  match values present in the training data.
- `class` as either `Economy` or `Business`. The backend encodes these as `0`
  and `1`.
- `stops` as `zero`, `one`, or `two_or_more`. The backend encodes these as
  `0`, `1`, and `2`.
- `duration` in hours. The training data range is `0.83` to `47.08` hours.
- `days_left` between `1` and `49`.

## Run with Docker

Build the image:

```powershell
docker build -t flight-price-api .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 flight-price-api
```

Then open:

```text
http://localhost:8000
```
