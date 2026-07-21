import json
import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
OUT_DIR = BASE / "outputs"

N_SAMPLE_ROWS = 300  

def main():
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    test_df = pd.read_parquet(DATA_DIR / "test_processed.parquet")
    sample = test_df.sample(min(N_SAMPLE_ROWS, len(test_df)), random_state=42)
    X_sample = sample[feature_cols]

   
    clf = joblib.load(MODEL_DIR / "failure_classifier.pkl")
    if hasattr(clf, "get_booster") or hasattr(clf, "estimators_"):
        explainer_clf = shap.TreeExplainer(clf)
    else:
        explainer_clf = shap.LinearExplainer(clf, X_sample)
    global_importance = {}
    if explainer_clf is not None:
        shap_values = explainer_clf.shap_values(X_sample)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:15]
        global_importance = {feature_cols[i]: round(float(mean_abs[i]), 4) for i in order}

    # --- Global feature importance for the regressor ---
    reg = joblib.load(MODEL_DIR / "rul_regressor.pkl")
    reg_importance = {}
    if hasattr(reg, "feature_importances_"):
        fi = reg.feature_importances_
        order = np.argsort(fi)[::-1][:15]
        reg_importance = {feature_cols[i]: round(float(fi[i]), 4) for i in order}

    # --- Per-engine explanation snapshot (most recent cycle per test engine) ---
    latest = test_df.sort_values("cycle").groupby("unit_id").tail(1)
    latest_X = latest[feature_cols]
    per_engine = []
    if explainer_clf is not None and len(latest_X) > 0:
        sv_latest = explainer_clf.shap_values(latest_X)
        sv_latest = sv_latest[1] if isinstance(sv_latest, list) else sv_latest
        for row_i, (_, row) in enumerate(latest.iterrows()):
            contribs = sv_latest[row_i]
            top_idx = np.argsort(np.abs(contribs))[::-1][:5]
            per_engine.append({
                "unit_id": int(row["unit_id"]),
                "top_factors": [
                    {"feature": feature_cols[i], "impact": round(float(contribs[i]), 4)}
                    for i in top_idx
                ],
            })

    out = {
        "classifier_global_importance": global_importance,
        "regressor_global_importance": reg_importance,
        "per_engine_explanations": per_engine,
    }
    with open(OUT_DIR / "explainability.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved explainability.json ({len(per_engine)} engine explanations)")
    print("Top classifier drivers:", list(global_importance.items())[:5])

if __name__ == "__main__":
    main()
