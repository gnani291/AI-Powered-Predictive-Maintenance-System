import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    mean_squared_error, roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
)
from xgboost import XGBRegressor, XGBClassifier

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

COST_FALSE_NEGATIVE = 50_000   # unplanned failure: downtime + damage + safety risk
COST_FALSE_POSITIVE = 1_200    # unnecessary inspection/maintenance visit

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(s))


def train_regression(X_train, y_train, X_test, y_test):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=9, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
    }
    results, preds = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = np.clip(pred, 0, None)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        score = nasa_score(y_test.values, pred)
        results[name] = {"rmse": round(rmse, 2), "nasa_score": round(score, 1)}
        preds[name] = pred
        print(f"[Regression] {name}: RMSE={rmse:.2f}  NASA-score={score:.1f}")

    best_name = min(results, key=lambda n: results[n]["rmse"])
    joblib.dump(models[best_name], MODEL_DIR / "rul_regressor.pkl")
    print(f"-> best regressor: {best_name}")
    return results, preds[best_name], best_name


def train_classification(X_train, y_train, X_test, y_test):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=60, max_depth=7, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            random_state=42, n_jobs=-1, eval_metric="logloss",
        ),
    }
    results, probs = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        roc = float(roc_auc_score(y_test, p))
        pr = float(average_precision_score(y_test, p))
        f1 = float(f1_score(y_test, (p >= 0.5).astype(int)))
        results[name] = {"roc_auc": round(roc, 3), "pr_auc": round(pr, 3), "f1_at_0.5": round(f1, 3)}
        probs[name] = p
        print(f"[Classification] {name}: ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}  F1@0.5={f1:.3f}")

    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    joblib.dump(models[best_name], MODEL_DIR / "failure_classifier.pkl")
    print(f"-> best classifier: {best_name}")
    return results, probs[best_name], best_name


def cost_sensitive_threshold(y_test, probs):
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        fn = int(((pred == 0) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        total_cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
        costs.append(total_cost)
    costs = np.array(costs)
    best_idx = int(np.argmin(costs))
    return {
        "best_threshold": round(float(thresholds[best_idx]), 3),
        "min_cost": int(costs[best_idx]),
        "cost_at_0.5": int(costs[np.argmin(np.abs(thresholds - 0.5))]),
        "sweep": [{"threshold": round(float(t), 2), "cost": int(c)} for t, c in zip(thresholds, costs)],
    }


def train_anomaly_detector(X_train, rul_train):
    healthy_mask = rul_train > 60
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_train[healthy_mask])
    joblib.dump(iso, MODEL_DIR / "anomaly_detector.pkl")
    return iso


def main():
    train_df = pd.read_parquet(DATA_DIR / "train_processed.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_processed.parquet")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    y_train_rul, y_test_rul = train_df["RUL"], test_df["RUL"]
    y_train_cls, y_test_cls = train_df["failure_imminent"], test_df["failure_imminent"]

    reg_results, best_rul_pred, best_reg_name = train_regression(X_train, y_train_rul, X_test, y_test_rul)
    cls_results, best_cls_prob, best_cls_name = train_classification(X_train, y_train_cls, X_test, y_test_cls)
    cost_analysis = cost_sensitive_threshold(y_test_cls.values, best_cls_prob)
    iso = train_anomaly_detector(X_train, y_train_rul)
    anomaly_scores_test = -iso.score_samples(X_test)  # higher = more anomalous
    anomaly_flags = iso.predict(X_test) == -1

    metrics = {
        "regression": reg_results,
        "best_regressor": best_reg_name,
        "classification": cls_results,
        "best_classifier": best_cls_name,
        "cost_analysis": {k: v for k, v in cost_analysis.items() if k != "sweep"},
        "cost_sweep": cost_analysis["sweep"],
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_engines_test": int(test_df["unit_id"].nunique()),
        "n_features": len(feature_cols),
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    export = test_df[["unit_id", "cycle", "RUL", "failure_imminent"]].copy()
    export["predicted_RUL"] = np.round(best_rul_pred, 1)
    export["failure_probability"] = np.round(best_cls_prob, 4)
    export["anomaly_score"] = np.round(anomaly_scores_test, 4)
    export["is_anomaly"] = anomaly_flags
    export.to_json(OUT_DIR / "predictions.json", orient="records")

    print("\nSaved metrics.json and predictions.json to outputs/")
    print(json.dumps({k: v for k, v in metrics.items() if k != "cost_sweep"}, indent=2))


if __name__ == "__main__":
    main()
