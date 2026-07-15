# TurbineWatch API

Optional FastAPI service for live inference, separate from the static dashboard.

## Run locally
```bash
pip install -r ../requirements.txt
uvicorn main:app --reload --app-dir .
```

## Endpoints
- `GET /health` — liveness check
- `POST /predict` — send a short window of recent cycles (raw sensor +
  operating-setting readings) for one engine, get back predicted RUL,
  failure probability, alert flag, and anomaly flag.

### Example request
```json
{
  "unit_id": 3,
  "cycles": [
    {"op_setting_1": -0.02, "op_setting_2": 0.001, "op_setting_3": 100.0,
     "sensor_1": 515.6, "sensor_2": 641.2, "...": "...through sensor_21"}
  ]
}
```
Send more than one cycle (ideally 10+) for accurate rolling-feature
computation — a single cycle works but rolling std/slope will be less
meaningful.

## Deploying
Any ASGI-compatible host works (Render, Railway, Fly.io, a small EC2/DO
box). Point requests at the `.pkl` files under `../models/` — make sure
those are committed or regenerated as part of your deploy step, since
they're the trained artifacts the API loads at startup.
