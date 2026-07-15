# TurbineWatch — AI-Powered Predictive Maintenance System

End-to-end predictive maintenance pipeline for a turbofan engine fleet:
RUL (Remaining Useful Life) regression, failure classification, unsupervised
anomaly detection, SHAP explainability, cost-sensitive decision thresholds,
and a live-monitoring style dashboard.

## Why this project

Most student CMAPSS projects stop at "train a classifier, report accuracy."
This one is built the way a real reliability-engineering team would frame it:

- **RUL regression**, not just binary failure — lets you rank engines by
  urgency, not just flag them.
- **NASA's asymmetric scoring function** — a late (optimistic) RUL prediction
  is penalized far more than an early one, because it's the one that causes
  unplanned downtime.
- **Cost-sensitive threshold selection** — the alert cutoff is chosen by
  minimizing (false-negative cost × count) + (false-positive cost × count),
  not defaulted to 0.5.
- **Unsupervised anomaly detection** layered on top of supervised models —
  catches sensor drift patterns that don't match any labeled failure mode.
- **SHAP explainability** — every prediction ships with "why," not just a number.

## Results (synthetic CMAPSS-style fleet, 100 engines, 87 features)

| Task | Best model | Headline metric |
|---|---|---|
| RUL regression | XGBoost | RMSE 17.4 cycles, NASA-score 29,027 |
| Failure classification (30-cycle horizon) | Logistic Regression | ROC-AUC 0.974, PR-AUC 0.802 |
| Cost-optimal alert threshold | — | 0.10 (vs. default 0.5) — cuts total fleet cost from $3.7M to $0.96M in the test window |

Full metrics: `outputs/metrics.json`. Per-cycle predictions: `outputs/predictions.json`.

## About the data

This sandbox has no network access to NASA's Prognostics Data Repository or
Kaggle, so `src/generate_data.py` **synthesizes** a CMAPSS-structured dataset:
21 sensors, 3 operating settings, 100 engines run to failure, with realistic
touches — fleet-consistent sensor drift direction, an accelerating
(convex) degradation curve near end-of-life, uninformative "flat" sensors
mixed in with real ones, and random sensor dropout that forces real cleaning.

**To use real CMAPSS data instead:** download `train_FD001.txt` from the
[NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/),
place it at `data/turbofan_raw.csv` with the same column layout
(`unit_id, cycle, op_setting_1..3, sensor_1..21`), and skip
`generate_data.py` — every other script works unmodified.

## Pipeline

```
src/generate_data.py     -> data/turbofan_raw.csv           (synthetic sensor traces)
src/preprocessing.py     -> data/{train,test}_processed.parquet, models/scaler.pkl
src/train_models.py      -> models/*.pkl, outputs/metrics.json, outputs/predictions.json
src/explainability.py    -> outputs/explainability.json     (SHAP)
```

Run in order:
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/preprocessing.py
python src/train_models.py
python src/explainability.py
```

### Feature engineering
Per-engine rolling window (10 cycles) mean / std / slope on every sensor,
on top of raw readings and operating settings — 87 features total. The
rolling slope is what actually carries the "is this degrading" signal;
raw instantaneous readings are noisy.

### Modeling detail
- **RUL label**: `min(max_cycle - cycle, 125)` — the standard piecewise-linear
  cap used across CMAPSS literature (early-life RUL is not meaningfully
  learnable from sensor data, so capping stabilizes training).
- **Classification label**: `1` if within 30 cycles of failure.
- **Anomaly detector**: Isolation Forest trained only on cycles with RUL > 60
  ("healthy" baseline), so it flags deviation from normal operation
  independent of the supervised failure label.

## Dashboard

`dashboard/index.html` is a **single self-contained file** — no build step,
no backend, all 20 held-out test engines' predictions embedded directly as
JSON. Open it in any browser, or deploy as a static site.

Panels:
- Fleet roster with health-tier color coding and anomaly flags
- Instrument-style RUL gauge (green/amber/red banded, like an EGT gauge)
- Actual vs. predicted RUL trend, failure probability trend
- SHAP driver bars for the selected engine's latest reading
- Interactive cost-sensitive threshold slider
- Model scoreboard (all candidate models, not just the winner)

### Deploying it
Since it's a single static HTML file:
```bash
# Vercel
vercel deploy dashboard/index.html

# or just drag dashboard/index.html into the Vercel/Netlify dashboard,
# or push the whole repo and set dashboard/ as the output directory,
# or open it directly — it works with no server at all.
```

To refresh the dashboard after retraining: re-run `src/train_models.py`,
`src/explainability.py`, then the export step in `src/export_dashboard_data.py`
(recomputes `dashboard/index.html`'s embedded data block).

## API (optional, for a live-inference story)

`api/main.py` is a FastAPI service that loads the trained `.pkl` models and
serves `/predict` for new sensor readings — useful if you want to demo
live inference rather than a static precomputed dashboard, or extend this
into a real deployed service. See `api/README.md`.

## Repo structure

```
predictive-maintenance-ai/
├── data/                       raw + processed data (generated, not committed)
├── models/                     trained model artifacts (generated)
├── outputs/                    metrics, predictions, SHAP export (generated)
├── src/
│   ├── generate_data.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── explainability.py
│   └── export_dashboard_data.py
├── api/
│   └── main.py                 FastAPI inference service
├── dashboard/
│   └── index.html              self-contained dashboard (deployable as-is)
├── requirements.txt
└── README.md
```

## Talking points for interviews

- "Why RMSE isn't enough" — walk through the NASA asymmetric scoring function
  and why late predictions cost more in the real world.
- "Why 0.5 threshold is wrong here" — the cost-sensitive sweep and how the
  optimal threshold (0.10) reflects the true asymmetry between a missed
  failure ($50K) and an unnecessary inspection ($1.2K).
- "How do you know the model isn't just memorizing engine IDs" — units are
  split at the engine level (`train_test_split_by_unit`), so the test set
  is entirely unseen engines, not just unseen cycles.
- "What does the anomaly detector add that the classifier doesn't" — it's
  unsupervised, trained only on healthy cycles, so it can flag a sensor
  pattern the model has never seen labeled, not just replay the training
  distribution of known failures.
