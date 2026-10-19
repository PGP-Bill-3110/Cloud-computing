import pandas as pd
import numpy as np
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)

# ==================================================
# 1. LOAD DATA
# ==================================================
DATA_PATH = "2025-C.csv"
df = pd.read_csv(DATA_PATH)

print("======================================")
print("Loading data completed")
print("======================================")

# ==================================================
# 2. FILTER BOD ATU
# ==================================================
df = df[df["determinand.label"] == "BOD ATU"].copy()
print(f"Total BOD ATU records (all stations): {len(df)}")

# ==================================================
# 3. RENAME COLUMNS
# ==================================================
df = df.rename(columns={
    "sample.samplingPoint.label": "station",
    "sample.sampleDateTime": "datetime",
    "sample.samplingPoint.easting": "easting",
    "sample.samplingPoint.northing": "northing",
    "result": "bod"
})

# ==================================================
# 4. TIME PROCESSING & SORT
# ==================================================
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

# ==================================================
# 5. CREATE LAG FEATURES (PER STATION)
# ==================================================
for lag in [1, 2, 3]:
    df[f"bod_lag_{lag}"] = df.groupby("station")["bod"].shift(lag)

df_model = df.dropna(
    subset=["bod_lag_1", "bod_lag_2", "bod_lag_3"]
).copy()

FEATURES = [
    "bod_lag_1",
    "bod_lag_2",
    "bod_lag_3",
    "easting",
    "northing"
]
TARGET = "bod"

stations = df_model["station"].unique()
total_stations = len(stations)

print("======================================")
print(f"Start evaluating {total_stations} stations")
print("======================================")

# ==================================================
# 6. ACCEPTABLE ERROR FUNCTION
# ==================================================
def acceptable_percentage(y_true, y_pred, tolerance):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    rel_error = np.abs(y_pred[mask] - y_true[mask]) / np.abs(y_true[mask])
    return np.mean(rel_error <= tolerance) * 100

# ==================================================
# 7. TRAIN & EVALUATE PER STATION
# ==================================================
station_metrics = []
start_all = time.time()

for idx, station in enumerate(stations, start=1):

    df_station_raw = df[df["station"] == station]
    df_station_model = df_model[df_model["station"] == station]

    num_bod_records_raw = len(df_station_raw)
    num_records_used = len(df_station_model)

    if num_records_used < 10:
        continue

    print(
        f"[{idx}/{total_stations}] Evaluating station: {station} | "
        f"BOD records: {num_bod_records_raw} | Used: {num_records_used}"
    )

    split_idx = int(num_records_used * 0.8)
    train = df_station_model.iloc[:split_idx]
    test = df_station_model.iloc[split_idx:]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    station_metrics.append({
        "Station": station,
        "Num_BOD_ATU_Records": num_bod_records_raw,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "MedianAE": median_absolute_error(y_test, y_pred),
        "Acceptable_10_percent": acceptable_percentage(y_test, y_pred, 0.10),
        "Acceptable_20_percent": acceptable_percentage(y_test, y_pred, 0.20)
    })

print(f"Total evaluation time: {time.time() - start_all:.2f} seconds")

# ==================================================
# 8. CREATE METRICS TABLE (SORT BY ACCEPTABLE_10%)
# ==================================================
station_metrics_df = pd.DataFrame(station_metrics)

station_metrics_df = station_metrics_df.sort_values(
    by="Acceptable_10_percent",
    ascending=False
).reset_index(drop=True)

# ==================================================
# 9. ACCURACY CLASSIFICATION (SAFE LABELS FOR EXCEL)
# ==================================================
def classify_accuracy_safe(value):
    if value == 0:
        return "0%"
    elif 0 < value <= 25:
        return "1%-25%"
    elif 25 < value <= 50:
        return "25%-50%"
    elif 50 < value <= 75:
        return "50%-75%"
    elif 75 < value < 100:
        return "75%-99%"
    elif value == 100:
        return "100%"
    else:
        return "Unknown"

station_metrics_df["AccuracyGroup_10"] = (
    station_metrics_df["Acceptable_10_percent"]
    .apply(classify_accuracy_safe)
)

station_metrics_df["AccuracyGroup_20"] = (
    station_metrics_df["Acceptable_20_percent"]
    .apply(classify_accuracy_safe)
)

# ==================================================
# 10. ACCURACY DISTRIBUTION TABLES
# ==================================================
bins_order = ["0%", "1%-25%", "25%-50%", "50%-75%", "75%-99%", "100%"]

accuracy_10_summary = (
    station_metrics_df["AccuracyGroup_10"]
    .value_counts()
    .reindex(bins_order, fill_value=0)
    .reset_index()
)
accuracy_10_summary.columns = [
    "Acceptable_10_percent_Range",
    "Number_of_Stations"
]

accuracy_20_summary = (
    station_metrics_df["AccuracyGroup_20"]
    .value_counts()
    .reindex(bins_order, fill_value=0)
    .reset_index()
)
accuracy_20_summary.columns = [
    "Acceptable_20_percent_Range",
    "Number_of_Stations"
]

# ==================================================
# 11. SAVE FILES (SORTED & EXCEL-SAFE)
# ==================================================
station_metrics_df.to_csv(
    "RF_BOD_ATU_metrics_all_stations_sorted.csv",
    index=False,
    encoding="utf-8-sig"
)

accuracy_10_summary.to_csv(
    "RF_BOD_ATU_accuracy_distribution_10_percent.csv",
    index=False,
    encoding="utf-8-sig"
)

accuracy_20_summary.to_csv(
    "RF_BOD_ATU_accuracy_distribution_20_percent.csv",
    index=False,
    encoding="utf-8-sig"
)

print("======================================")
print("Files saved successfully (sorted & Excel-safe):")
print("- RF_BOD_ATU_metrics_all_stations_sorted.csv")
print("- RF_BOD_ATU_accuracy_distribution_10_percent.csv")
print("- RF_BOD_ATU_accuracy_distribution_20_percent.csv")
print("======================================")
