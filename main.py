from pathlib import Path
from typing import Any
from html import escape

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).with_name("model.pkl")


def load_model() -> Any:
    """Load the model once when the application starts."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Could not load model.pkl. Ensure it was saved with joblib and "
            "that the local scikit-learn version matches the training version."
        ) from exc


model = load_model()
MODEL_LOADED = model is not None
MODEL_FEATURES = [str(feature) for feature in model.feature_names_in_]
FEATURE_INDEX = {name: index for index, name in enumerate(MODEL_FEATURES)}


def model_categories(field: str) -> list[str]:
    prefix = f"__{field}_"
    return sorted(
        name[len(prefix):]
        for name in MODEL_FEATURES
        if name.startswith(prefix)
    )


AIRLINES = model_categories("airline")
SOURCES = model_categories("source")
DESTINATIONS = model_categories("destination")
DEPARTURE_TIMES = model_categories("departure")
ARRIVAL_TIMES = model_categories("arrival")
CLASS_VALUES = ["Economy", "Business"]
STOPS_VALUES = ["zero", "one", "two_or_more"]
CLASS_ENCODING = {"Economy": 0, "Business": 1}
STOPS_ENCODING = {"zero": 0, "one": 1, "two_or_more": 2}
DEFAULT_FLIGHT = (model_categories("flight") or ["AI-401"])[0]


def option_tags(values: list[Any]) -> str:
    return "".join(
        f'<option value="{escape(str(value))}">{escape(str(value).replace("_", " "))}</option>'
        for value in values
    )


class FlightFeatures(BaseModel):
    """Raw flight features used to build the model's one-hot encoded row."""

    model_config = ConfigDict(populate_by_name=True)

    airline: str = Field(..., examples=["Indigo"])
    source: str = Field(..., examples=["Delhi"])
    departure: str = Field(..., examples=["Morning"])
    arrival: str = Field(..., examples=["Evening"])
    destination: str = Field(..., examples=["Mumbai"])
    # These fields match the features used by the retrained model.
    flight: str = Field(DEFAULT_FLIGHT, examples=[DEFAULT_FLIGHT])
    stops: str = Field("one", examples=["one"])
    travel_class: str = Field("Economy", alias="class", examples=["Business"])
    duration: float = Field(12.0, ge=0.83, le=47.08, examples=[12.0])
    days_left: float = Field(26, ge=1, le=49, examples=[26])


class PredictionResponse(BaseModel):
    predicted_price: float


app = FastAPI(
    title="Flight Price Prediction API",
    description="Predict flight prices using the trained Decision Tree regression model.",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> str:
    page = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Flight Price Predictor</title>
  <style>
    :root { --ink:#132238; --muted:#6d7b8f; --line:#dfe7ef; --blue:#2563eb; --blue-dark:#1745a5; --cream:#f5f8fc; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color:var(--ink); background:linear-gradient(135deg,#eef5ff 0%,#f9fbfd 48%,#eaf7f3 100%); }
    .shell { width:min(1120px, calc(100% - 32px)); margin:0 auto; padding:42px 0 56px; }
    .topbar { display:flex; justify-content:space-between; align-items:center; gap:20px; margin-bottom:48px; }
    .brand { display:flex; align-items:center; gap:12px; font-weight:800; letter-spacing:-.03em; }
    .brand-mark { width:38px; height:38px; display:grid; place-items:center; border-radius:12px; background:var(--ink); color:#fff; font-size:20px; transform:rotate(-8deg); }
    .docs { display:none; }
    .hero { max-width:700px; margin-bottom:30px; }
    .eyebrow { color:var(--blue); text-transform:uppercase; letter-spacing:.16em; font-size:12px; font-weight:800; }
    h1 { font-size:clamp(38px,6vw,68px); line-height:.98; letter-spacing:-.065em; margin:12px 0 18px; }
    .hero p { color:var(--muted); font-size:18px; line-height:1.55; max-width:600px; margin:0; }
    .route-line { display:flex; align-items:center; gap:12px; max-width:440px; margin:26px 0 0; color:#9aa9bb; font-size:12px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
    .route-line::before, .route-line::after { content:""; height:1px; flex:1; background:linear-gradient(90deg,transparent,#b8c6d6); }
    .route-line::after { background:linear-gradient(90deg,#b8c6d6,transparent); }
    .route-plane { color:var(--blue); font-size:18px; transform:rotate(8deg); }
    .layout { display:grid; grid-template-columns:minmax(0,1.5fr) minmax(280px,.8fr); gap:22px; align-items:stretch; }
    .card { background:rgba(255,255,255,.9); border:1px solid rgba(223,231,239,.9); border-radius:24px; box-shadow:0 18px 50px rgba(24,52,84,.08); padding:28px; transition:transform .2s ease, box-shadow .2s ease; }
    .card:hover { transform:translateY(-2px); box-shadow:0 22px 58px rgba(24,52,84,.12); }
    .card h2 { margin:0 0 22px; font-size:20px; letter-spacing:-.03em; }
    .form-intro { color:var(--muted); margin:-10px 0 22px; font-size:14px; line-height:1.5; }
    .form-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
    label { display:grid; gap:7px; color:var(--muted); font-size:13px; font-weight:700; }
    input, select { width:100%; border:1px solid var(--line); border-radius:12px; padding:12px 13px; background:#fff; color:var(--ink); font:inherit; outline:none; }
    input:focus, select:focus { border-color:var(--blue); box-shadow:0 0 0 3px rgba(37,99,235,.12); }
    .full { grid-column:1 / -1; }
    button { width:100%; border:0; border-radius:13px; padding:14px 18px; margin-top:20px; background:var(--blue); color:#fff; font:inherit; font-weight:800; cursor:pointer; transition:.2s ease; }
    button:hover { background:var(--blue-dark); transform:translateY(-1px); }
    button:disabled { opacity:.65; cursor:wait; transform:none; }
    .result { min-height:100%; display:flex; flex-direction:column; justify-content:space-between; background:var(--ink); color:#fff; }
    .result h2 { color:#fff; }
    .result-copy { color:#afbed2; line-height:1.6; }
    .price { margin:42px 0 10px; font-size:clamp(38px,5vw,58px); letter-spacing:-.06em; font-weight:850; color:#a7f3d0; overflow-wrap:anywhere; }
    .price-label { color:#afbed2; font-size:13px; text-transform:uppercase; letter-spacing:.12em; font-weight:800; }
    .status { margin-top:18px; color:#fca5a5; font-size:14px; line-height:1.45; }
    .tip { border-top:1px solid rgba(255,255,255,.14); padding-top:18px; color:#afbed2; font-size:13px; line-height:1.5; }
    @media (max-width:760px) { .topbar { margin-bottom:32px; } .layout { grid-template-columns:1fr; } .form-grid { grid-template-columns:1fr; } .full { grid-column:auto; } }
  </style>
</head>
<body>
  <main class="shell">
    <nav class="topbar">
      <div class="brand"><span class="brand-mark">✈</span><span>Farewise</span></div>
      <a class="docs" href="/docs" target="_blank">API documentation ↗</a>
    </nav>
    <section class="hero">
      <div class="eyebrow">Machine learning · flight fares</div>
      <h1>Know the fare before you fly.</h1>
      <p>Enter the details of your journey and let the trained model estimate the ticket price.</p>
      <div class="route-line"><span>your trip</span><span class="route-plane">✈</span><span>your estimate</span></div>
    </section>
    <section class="layout">
      <form class="card" id="prediction-form">
        <h2>Trip details</h2>
        <p class="form-intro">Choose your journey details below to get a live prediction.</p>
        <div class="form-grid">
          <label>Airline<select name="airline" required>__AIRLINE_OPTIONS__</select></label>
          <label>From<select name="source" required>__SOURCE_OPTIONS__</select></label>
          <label>To<select name="destination" required>__DESTINATION_OPTIONS__</select></label>
          <label>Departure time<select name="departure" required>__DEPARTURE_OPTIONS__</select></label>
          <label>Arrival time<select name="arrival" required>__ARRIVAL_OPTIONS__</select></label>
          <label>Class<select name="class" required>__CLASS_OPTIONS__</select></label>
          <label>Flight<select name="flight" required>__FLIGHT_OPTIONS__</select></label>
          <label>Stops<select name="stops" required>__STOPS_OPTIONS__</select></label>
          <label>Duration (hours)<input name="duration" type="number" min="0.83" max="47.08" step="0.01" value="12.00" required /></label>
          <label>Days until departure<input name="days_left" type="number" min="1" max="49" step="1" value="26" required /></label>
        </div>
        <button id="submit-button" type="submit">Predict flight price</button>
      </form>
      <aside class="card result">
        <div>
          <h2>Estimated fare</h2>
          <p class="result-copy">Your prediction will appear here after you submit the trip details.</p>
          <div class="price" id="price">—</div>
          <div class="price-label">Predicted price</div>
          <div class="status" id="status" role="alert"></div>
        </div>
        <p class="tip">Tip: categorical values such as airline, city, and flight number must match values present in the training data.</p>
      </aside>
    </section>
  </main>
  <script>
    const form = document.getElementById('prediction-form');
    const button = document.getElementById('submit-button');
    const price = document.getElementById('price');
    const status = document.getElementById('status');

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      button.textContent = 'Predicting…';
      status.textContent = '';
      price.textContent = '…';

      const values = Object.fromEntries(new FormData(form).entries());
      for (const key of ['duration', 'days_left']) {
        values[key] = Number(values[key]);
      }

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(values)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Prediction failed.');
        price.textContent = Number(data.predicted_price).toLocaleString(undefined, {maximumFractionDigits: 2});
      } catch (error) {
        price.textContent = '—';
        status.textContent = error.message;
      } finally {
        button.disabled = false;
        button.textContent = 'Predict flight price';
      }
    });
  </script>
</body>
</html>
"""
    return (
        page
        .replace("__AIRLINE_OPTIONS__", option_tags(AIRLINES))
        .replace("__SOURCE_OPTIONS__", option_tags(SOURCES))
        .replace("__DESTINATION_OPTIONS__", option_tags(DESTINATIONS))
        .replace("__DEPARTURE_OPTIONS__", option_tags(DEPARTURE_TIMES))
        .replace("__ARRIVAL_OPTIONS__", option_tags(ARRIVAL_TIMES))
        .replace("__CLASS_OPTIONS__", option_tags(CLASS_VALUES))
        .replace("__STOPS_OPTIONS__", option_tags(STOPS_VALUES))
        .replace("__FLIGHT_OPTIONS__", option_tags(model_categories("flight")))
    )


@app.get("/health", tags=["health"])
def health() -> dict[str, str | bool]:
    return {
        "status": "ok" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
    }


@app.head("/", include_in_schema=False)
def root_head() -> Response:
    return Response(status_code=200)


def build_model_row(features: FlightFeatures) -> np.ndarray:
    values = features.model_dump(by_alias=False)
    row = np.zeros((1, len(MODEL_FEATURES)), dtype=float)

    if values["travel_class"] not in CLASS_ENCODING:
        raise HTTPException(status_code=422, detail="class must be Economy or Business")
    if values["stops"] not in STOPS_ENCODING:
        raise HTTPException(status_code=422, detail="stops must be zero, one, or two_or_more")

    categorical_fields = ("airline", "source", "departure", "arrival", "destination", "flight")
    for field in categorical_fields:
        feature_name = f"__{field}_{values[field]}"
        if feature_name not in FEATURE_INDEX:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unknown {field} value {values[field]!r}. "
                    "The value was not present in the training data."
                ),
            )
        row[0, FEATURE_INDEX[feature_name]] = 1.0

    numeric_fields = ("stops", "class", "duration", "days_left")
    for field in numeric_fields:
        feature_name = f"remainder__{field}"
        if feature_name not in FEATURE_INDEX:
            raise HTTPException(status_code=500, detail=f"Model is missing feature {feature_name}")
        if field == "class":
            value = CLASS_ENCODING[values["travel_class"]]
        elif field == "stops":
            value = STOPS_ENCODING[values["stops"]]
        else:
            value = values[field]
        row[0, FEATURE_INDEX[feature_name]] = value

    return row


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(features: FlightFeatures) -> PredictionResponse:
    row = build_model_row(features)
    try:
        prediction = float(np.asarray(model.predict(row)).reshape(-1)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return PredictionResponse(predicted_price=prediction)
