import pandas as pd
import numpy as np
import math
import time
from datetime import datetime

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)

# ==================================================
# CONFIG
# ==================================================
DATA_PATH = "2025-C.csv"
MIN_RECORDS_PER_STATION = 10

# ==================================================
# WELFORD NORMALIZATION (ONLINE)
# ==================================================
class WelfordScaler:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.count += 1
        if self.count == 1:
            self.mean = x
            self.M2 = 0.0
        else:
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def std(self):
        if self.count < 2:
            return 1.0
        return math.sqrt(self.M2 / (self.count - 1))


# ==================================================
# ACCEPTABLE ERROR
# ==================================================
def acceptable_percentage(y_true, y_pred, tolerance):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    rel_error = np.abs(y_pred[mask] - y_true[mask]) / np.abs(y_true[mask])
    return np.mean(rel_error <= tolerance) * 100


# ==================================================
# LOAD & FILTER DATA
# ==================================================
df = pd.read_csv(DATA_PATH)

df = df[df["determinand.label"] == "BOD ATU"].copy()

df = df.rename(columns={
    "sample.samplingPoint.label": "station",
    "sample.sampleDateTime": "datetime",
    "result": "bod"
})

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

print(f"Total BOD ATU records: {len(df)}")

# ==================================================
# ONLINE EVALUATION PER STATION
# ==================================================
station_results = []
start_all = time.time()

stations = df["station"].unique()

for idx, station in enumerate(stations, start=1):

    g = df[df["station"] == station]
    if len(g) < MIN_RECORDS_PER_STATION:
        continue

    print(f"[{idx}/{len(stations)}] Evaluating ONLINE station: {station}")

    # Online model
    model = SGDRegressor(
        loss="squared_error",
        learning_rate="adaptive",
        eta0=0.01,
        random_state=42
    )

    scaler = WelfordScaler()
    initialized = False

    y_true_all = []
    y_pred_all = []

    # PREQUENTIAL EVALUATION
    for _, row in g.iterrows():
        ts = row["datetime"].timestamp()
        value = row["bod"]

        scaler.update(ts)
        std = max(scaler.std(), 1e-6)
        X = np.array([[(ts - scaler.mean) / std]])
        y = np.array([value])

        if initialized:
            y_pred = model.predict(X)[0]
            y_pred_all.append(y_pred)
            y_true_all.append(value)

            model.partial_fit(X, y)
        else:
            model.partial_fit(X, y)
            initialized = True

    # Skip nếu không đủ điểm để đánh giá
    if len(y_true_all) < 3:
        continue

    # METRICS
    mae = mean_absolute_error(y_true_all, y_pred_all)
    mse = mean_squared_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_all, y_pred_all)
    medae = median_absolute_error(y_true_all, y_pred_all)

    acc10 = acceptable_percentage(y_true_all, y_pred_all, 0.10)
    acc20 = acceptable_percentage(y_true_all, y_pred_all, 0.20)

    station_results.append({
        "Station": station,
        "Num_BOD_ATU_Records": len(g),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MedianAE": medae,
        "Acceptable_10_percent": acc10,
        "Acceptable_20_percent": acc20
    })

print(f"Total online evaluation time: {time.time() - start_all:.2f} seconds")

# ==================================================
# CREATE RESULT TABLES
# ==================================================
station_metrics_df = pd.DataFrame(station_results)

station_metrics_df = station_metrics_df.sort_values(
    by="Acceptable_10_percent",
    ascending=False
).reset_index(drop=True)

# ==================================================
# SAVE FILES
# ==================================================
station_metrics_df.to_csv(
    "ONLINE_SGD_BOD_ATU_metrics_all_stations_sorted.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Saved ONLINE SGD evaluation results.")
