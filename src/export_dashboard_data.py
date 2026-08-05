import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "outputs"
DASHBOARD_HTML = BASE / "dashboard" / "index.html"

DATA_MARKER_START = "window.__DASHBOARD_DATA__ = "
DATA_MARKER_END = ";</script>"

def build_payload():
    preds = pd.DataFrame(json.load(open(OUT_DIR / "predictions.json")))
    metrics = json.load(open(OUT_DIR / "metrics.json"))
    explain = json.load(open(OUT_DIR / "explainability.json"))

    engines = {}
    for uid, g in preds.groupby("unit_id"):
        g = g.sort_values("cycle")
        engines[int(uid)] = {
            "cycles": g["cycle"].tolist(),
            "rul": g["RUL"].tolist(),
            "predicted_rul": g["predicted_RUL"].round(1).tolist(),
            "failure_prob": g["failure_probability"].round(3).tolist(),
            "anomaly_score": g["anomaly_score"].round(3).tolist(),
            "is_anomaly": g["is_anomaly"].tolist(),
            "latest_cycle": int(g["cycle"].max()),
            "latest_rul_actual": int(g["RUL"].iloc[-1]),
            "latest_rul_pred": float(g["predicted_RUL"].iloc[-1]),
            "latest_failure_prob": float(g["failure_probability"].iloc[-1]),
            "latest_anomaly": bool(g["is_anomaly"].iloc[-1]),
        }

    top_drivers = {int(e["unit_id"]): e["top_factors"] for e in explain["per_engine_explanations"]}

    return {
        "metrics": {k: v for k, v in metrics.items() if k != "cost_sweep"},
        "cost_sweep": metrics["cost_sweep"][::4],
        "engines": engines,
        "global_importance_classifier": explain["classifier_global_importance"],
        "global_importance_regressor": explain["regressor_global_importance"],
        "top_drivers": top_drivers,
    }


def main():
    payload = build_payload()
    with open(OUT_DIR / "dashboard_data.json", "w") as f:
        json.dump(payload, f)

    html = DASHBOARD_HTML.read_text()
    start = html.find(DATA_MARKER_START)
    end = html.find(DATA_MARKER_END, start)
    if start == -1 or end == -1:
        raise RuntimeError("Could not find embedded data block in dashboard/index.html")
    new_html = (
        html[:start] + DATA_MARKER_START + json.dumps(payload) + html[end:]
    )
    DASHBOARD_HTML.write_text(new_html)
    print(f"Refreshed dashboard with {len(payload['engines'])} engines' data.")

if __name__ == "__main__":
    main()
