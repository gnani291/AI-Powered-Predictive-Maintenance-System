
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

RUL_CAP = 125
ROLLING_WINDOW = 10
FAILURE_HORIZON = 30  # cycles-to-failure threshold for the classification task

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    df[sensor_cols] = df.groupby("unit_id")[sensor_cols].ffill().bfill()
    df[sensor_cols] = df[sensor_cols].fillna(df[sensor_cols].median())
    return df


def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    raw_rul = max_cycle - df["cycle"]
    df["RUL"] = raw_rul.clip(upper=RUL_CAP)
    df["failure_imminent"] = (raw_rul <= FAILURE_HORIZON).astype(int)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("unit_id")[sensor_cols]

    roll_mean = grouped.transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())
    roll_std = grouped.transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0))

    def slope(s):
        idx = np.arange(len(s))
        if len(s) < 2:
            return pd.Series(np.zeros(len(s)), index=s.index)
        # rolling linear-regression slope, vectorized via rolling apply
        return s.rolling(ROLLING_WINDOW, min_periods=2).apply(
            lambda w: np.polyfit(np.arange(len(w)), w, 1)[0] if len(w) >= 2 else 0.0,
            raw=True,
        ).fillna(0)

    roll_slope = grouped.transform(slope)

    roll_mean.columns = [f"{c}_rollmean" for c in sensor_cols]
    roll_std.columns = [f"{c}_rollstd" for c in sensor_cols]
    roll_slope.columns = [f"{c}_rollslope" for c in sensor_cols]

    out = pd.concat([df, roll_mean, roll_std, roll_slope], axis=1)
    return out


def build_feature_matrix(df: pd.DataFrame):
    sensor_cols = [c for c in df.columns if c.startswith("sensor_") and "_roll" not in c]
    setting_cols = [c for c in df.columns if c.startswith("op_setting")]
    engineered_cols = [c for c in df.columns if c.endswith(("_rollmean", "_rollstd", "_rollslope"))]
    feature_cols = sensor_cols + setting_cols + engineered_cols
    return df, feature_cols


def train_test_split_by_unit(df: pd.DataFrame, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    units = df["unit_id"].unique()
    rng.shuffle(units)
    n_test = int(len(units) * test_frac)
    test_units = set(units[:n_test])
    train_df = df[~df["unit_id"].isin(test_units)].copy()
    test_df = df[df["unit_id"].isin(test_units)].copy()
    return train_df, test_df


def run_pipeline():
    df = load_and_clean(DATA_DIR / "turbofan_raw.csv")
    df = add_rul_labels(df)
    df = add_rolling_features(df)
    df, feature_cols = build_feature_matrix(df)

    train_df, test_df = train_test_split_by_unit(df)

    scaler = StandardScaler()
    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

    train_df.to_parquet(DATA_DIR / "train_processed.parquet")
    test_df.to_parquet(DATA_DIR / "test_processed.parquet")

    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,} | Features: {len(feature_cols)}")
    print(f"Failure-imminent rate (train): {train_df['failure_imminent'].mean():.3f}")
    return train_df, test_df, feature_cols


if __name__ == "__main__":
    run_pipeline()
