import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_ENGINES = 100
N_SENSORS = 21
SENSOR_NAMES = [f"sensor_{i+1}" for i in range(N_SENSORS)]
SETTING_NAMES = ["op_setting_1", "op_setting_2", "op_setting_3"]

# Which sensors actually degrade meaningfully (mirrors CMAPSS: several
# sensors are flat/uninformative, which is itself a realistic modeling trap).
DEGRADING_SENSORS = [2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20]

# Each degrading sensor has a FIXED drift direction across the whole fleet
# (e.g. bearing-temp sensors always trend up as bearings wear) -- this is
# what makes cross-engine pooled models learnable in real turbofan data.
SENSOR_DIRECTION = {s: RNG.choice([-1, 1]) for s in DEGRADING_SENSORS}
SENSOR_STRENGTH = {s: RNG.uniform(8, 22) for s in DEGRADING_SENSORS}


SENSOR_BASE = {s: RNG.uniform(280, 650) for s in range(1, N_SENSORS + 1)}

def simulate_engine(unit_id: int) -> pd.DataFrame:
    life = int(RNG.integers(140, 360))  # total cycles until failure
    t = np.arange(1, life + 1)

    # Operating settings: mostly stable with small process noise
    settings = RNG.normal(loc=[0.0, 0.0, 100.0], scale=[0.02, 0.002, 5.0], size=(life, 3))

    sensors = np.zeros((life, N_SENSORS))
    frac_life = t / life  # 0 -> 1, used to shape degradation curve

    for s in range(N_SENSORS):
        sensor_id = s + 1
        base = SENSOR_BASE[sensor_id] * RNG.uniform(0.97, 1.03)  # slight unit-to-unit calibration variance
        noise = RNG.normal(0, 1, size=life)
        if sensor_id in DEGRADING_SENSORS:
            # per-engine variation in HOW FAST it degrades, but direction
            # and rough magnitude are consistent fleet-wide (learnable signal)
            drift_strength = SENSOR_STRENGTH[sensor_id] * RNG.uniform(0.75, 1.25)
            # accelerating (convex) degradation curve, steeper near end-of-life
            curve = drift_strength * (frac_life ** 2.2)
            direction = SENSOR_DIRECTION[sensor_id]
            sensors[:, s] = base + direction * curve + noise * (base * 0.0025)
        else:
            sensors[:, s] = base + noise * (base * 0.003)

    df = pd.DataFrame(sensors, columns=SENSOR_NAMES)
    df.insert(0, "cycle", t)
    df.insert(0, "unit_id", unit_id)
    for i, col in enumerate(SETTING_NAMES):
        df[col] = settings[:, i]

    # Occasional sensor dropout / missing values -> forces real cleaning work
    mask = RNG.random(size=df[SENSOR_NAMES].shape) < 0.003
    df[SENSOR_NAMES] = df[SENSOR_NAMES].mask(mask)

    return df


def main():
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(exist_ok=True)

    frames = [simulate_engine(uid) for uid in range(1, N_ENGINES + 1)]
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out_dir / "turbofan_raw.csv", index=False)
    print(f"Generated {len(full):,} rows across {N_ENGINES} engines -> {out_dir/'turbofan_raw.csv'}")
    print(full.head())


if __name__ == "__main__":
    main()
