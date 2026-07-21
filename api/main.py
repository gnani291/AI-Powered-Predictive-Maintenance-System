from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models"

app = FastAPI(title="TurbineWatch Predictive Maintenance API", version="1.0")

_scaler = joblib.load(MODEL_DIR / "scaler.pkl")
_feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
_rul_model = joblib.load(MODEL_DIR / "rul_regressor.pkl")
_cls_model = joblib.load(MODEL_DIR / "failure_classifier.pkl")
_anomaly_model = joblib.load(MODEL_DIR / "anomaly_detector.pkl")

ALERT_THRESHOLD = 0.10  # from the cost-sensitive sweep in train_models.py


class CycleWindow(BaseModel):
    """A short history of recent cycles for one engine, most recent last.
    Needs >=1 row; rolling features are more accurate with >=10."""
    unit_id: int
    cycles: list[dict] = Field(
        ...,
        description="Each dict must have keys: op_setting_1..3, sensor_1..21",
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: CycleWindow):
    if not payload.cycles:
        raise HTTPException(400, "cycles must be non-empty")

    df = pd.DataFrame(payload.cycles)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    if not sensor_cols:
        raise HTTPException(400, "no sensor_* columns found in input")

    window = 10
    roll_mean = df[sensor_cols].rolling(window, min_periods=1).mean()
    roll_std = df[sensor_cols].rolling(window, min_periods=1).std().fillna(0)
    roll_slope = df[sensor_cols].rolling(window, min_periods=2).apply(
        lambda w: np.polyfit(np.arange(len(w)), w, 1)[0] if len(w) >= 2 else 0.0, raw=True
    ).fillna(0)
    roll_mean.columns = [f"{c}_rollmean" for c in sensor_cols]
    roll_std.columns = [f"{c}_rollstd" for c in sensor_cols]
    roll_slope.columns = [f"{c}_rollslope" for c in sensor_cols]

    features = pd.concat([df, roll_mean, roll_std, roll_slope], axis=1)
    latest = features.iloc[[-1]]

    missing = [c for c in _feature_cols if c not in latest.columns]
    if missing:
        raise HTTPException(400, f"missing required fields: {missing}")

    X = latest[_feature_cols]
    X_scaled = _scaler.transform(X)

    rul = float(np.clip(_rul_model.predict(X_scaled)[0], 0, None))
    fail_prob = float(_cls_model.predict_proba(X_scaled)[0, 1])
    anomaly = bool(_anomaly_model.predict(X_scaled)[0] == -1)

    return {
        "unit_id": payload.unit_id,
        "predicted_rul_cycles": round(rul, 1),
        "failure_probability": round(fail_prob, 4),
        "alert": fail_prob >= ALERT_THRESHOLD,
        "alert_threshold_used": ALERT_THRESHOLD,
        "is_anomaly": anomaly,
    }
