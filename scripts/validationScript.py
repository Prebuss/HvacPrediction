import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

##########################################
# 1) CONFIGURATION / PATHS
##########################################

DATA_FOLDER = "/home/preben/Documents/Master/"  # Same as last script
VAL_YR_FOLDER = os.path.join(DATA_FOLDER, "selectiveYr") # Actually valYr
VAL_VENT_FOLDER = os.path.join(DATA_FOLDER, "selectiveVent") # Actually valVent
VAL_ADAX_FOLDER = os.path.join(DATA_FOLDER, "selectiveAdax") # Actually valAdax
VAL_QLARM_FOLDER = os.path.join(DATA_FOLDER, "selectiveQlarm") # Actually valQlarm

# Name for merged validation CSV (optional)
VAL_MERGED_CSV = "merged_val_hvac_dataset.csv"

SAVED_MODEL_PATH = "lstm_hvac_forecast_best_random.keras"
SAVED_SCALER_PATH = "min_max_scaler.pkl"

# Must match training script
FEATURES = [
    'co2',
    'supplyAirflow',
    'extractAirflow',
    'supplyAirFanSpeedLevel',
    'extractAirFanSpeedLevel',
    'SFP',
    'supplyAirTemperature',
    'extractAirTemperature',
    'outdoorTemperature',
    'supplyAirFilterPressure',
    'extractAirFilterPressure',
    'heatExchangerRegulator',
    'setpointTemperatureSA',
    'setpointTemperatureEA',
    'air_temperature',
    'dew_point_temperature',
    'relative_humidity',
    'wind_speed',
    'precipitation_amount',
    'Preben - Office Average - Temperature'
]
TARGET = "target_temp"
SEQ_LENGTH = 24  # same as training

##########################################
# 2) MERGING LOGIC FOR VALIDATION FILES
##########################################

def load_json_files(folder_path):
    """Utility to load all .json files in a given folder."""
    data_list = []
    if not os.path.isdir(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                data_list.append(json.load(f))
    return data_list

def merge_val_yr():
    dfs = []
    if os.path.isdir(VAL_YR_FOLDER):
        for file in sorted(os.listdir(VAL_YR_FOLDER)):
            if file.endswith(".json"):
                file_path = os.path.join(VAL_YR_FOLDER, file)
                date_str = file[2:12]  # e.g., "yr2025-03-09.json" -> "2025-03-09"
                with open(file_path, "r") as f:
                    data = json.load(f)
                df = pd.DataFrame([
                    {
                        "timestamp": f"{date_str} {entry['time']}",
                        **entry["data"]
                    }
                    for entry in data
                ])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                dfs.append(df)
    if dfs:
        out = pd.concat(dfs, ignore_index=True)
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out
    return pd.DataFrame()

def merge_val_vent():
    dfs = []
    if os.path.isdir(VAL_VENT_FOLDER):
        vent_data = load_json_files(VAL_VENT_FOLDER)
        for data in vent_data:
            df = pd.DataFrame(data)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                dfs.append(df)
    if dfs:
        out = pd.concat(dfs, ignore_index=True)
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out
    return pd.DataFrame()

def merge_val_adax():
    dfs = []
    if os.path.isdir(VAL_ADAX_FOLDER):
        adax_data = load_json_files(VAL_ADAX_FOLDER)
        for data in adax_data:
            temp_list = []
            for entry in data:
                timestamp = entry.get("timestamp")
                for room in entry.get("rooms", []):
                    energy_sum = sum(d["energy"] for d in room.get("devices", []))
                    temp_list.append({
                        "timestamp": timestamp,
                        "room_id": room.get("id"),
                        "room_name": room.get("name"),
                        "target_temp": room.get("targetTemperature"),
                        "current_temp": room.get("currentTemperature"),
                        "energy_consumption": energy_sum
                    })
            df = pd.DataFrame(temp_list)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                dfs.append(df)
    if dfs:
        out = pd.concat(dfs, ignore_index=True)
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out
    return pd.DataFrame()

def merge_val_qlarm():
    dfs = []
    if os.path.isdir(VAL_QLARM_FOLDER):
        for file in os.listdir(VAL_QLARM_FOLDER):
            if file.endswith(".csv"):
                file_path = os.path.join(VAL_QLARM_FOLDER, file)
                df = pd.read_csv(file_path)
                if "Time" in df.columns:
                    df.rename(columns={"Time": "timestamp"}, inplace=True)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                # pivot if needed
                df = df.pivot_table(
                    index="timestamp",
                    columns="SensorName",
                    values="Value",
                    aggfunc="mean"
                ).reset_index()
                dfs.append(df)
    if dfs:
        out = pd.concat(dfs, ignore_index=True)
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out
    return pd.DataFrame()

def merge_validation_data():
    """Merge valYr, valVent, valAdax, valQlarm -> single DF -> forward-fill -> compute office avg temperature -> save CSV."""
    print("🔄 Merging validation data from multiple folders...")

    yr_df = merge_val_yr()
    vent_df = merge_val_vent()
    adax_df = merge_val_adax()
    qlarm_df = merge_val_qlarm()

    print("valYr shape:", yr_df.shape)
    print("valVent shape:", vent_df.shape)
    print("valAdax shape:", adax_df.shape)
    print("valQlarm shape:", qlarm_df.shape)

    merged_df = (
        adax_df.merge(vent_df, on="timestamp", how="outer")
               .merge(yr_df, on="timestamp", how="outer")
               .merge(qlarm_df, on="timestamp", how="outer")
    )
    merged_df.sort_values("timestamp", inplace=True)
    merged_df.ffill(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    print(f"Validation merged DataFrame shape: {merged_df.shape}")

    # Compute "Preben - Office Average - Temperature"
    qlarm_temperature_cols = [
        c for c in merged_df.columns
        if "Temperature" in c and c != "Preben - Office Average - Temperature"
    ]
    if qlarm_temperature_cols:
        merged_df["Preben - Office Average - Temperature"] = merged_df[qlarm_temperature_cols].mean(axis=1)
        print(f"Computed 'Preben - Office Average - Temperature' from {len(qlarm_temperature_cols)} column(s).")
    else:
        print("⚠️ No Qlarm temperature columns found for 'Preben - Office Average - Temperature'.")

    # Save CSV for reference
    merged_df.to_csv(VAL_MERGED_CSV, index=False)
    print(f"✅ Validation merged CSV saved to '{VAL_MERGED_CSV}' with {len(merged_df)} rows.")

    return merged_df

##########################################
# 3) CREATE SEQUENCES & EVALUATE
##########################################

def create_sequences(data, seq_length, num_features, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq_X = data[i : i + seq_length, :num_features]
        seq_y = data[i + seq_length, target_idx]
        if not np.isnan(seq_X).any() and not np.isnan(seq_y):
            X.append(seq_X)
            y.append(seq_y)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def main():
    ##########################################
    # 1) MERGE VALIDATION DATA
    ##########################################
    val_df = merge_validation_data()

    ##########################################
    # 2) LOAD SCALER & MODEL
    ##########################################
    if not os.path.exists(SAVED_SCALER_PATH):
        raise FileNotFoundError(f"Scaler file '{SAVED_SCALER_PATH}' not found.")
    with open(SAVED_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print(f"✅ Loaded scaler from '{SAVED_SCALER_PATH}'.")

    if not os.path.exists(SAVED_MODEL_PATH):
        raise FileNotFoundError(f"Model file '{SAVED_MODEL_PATH}' not found.")
    model = load_model(SAVED_MODEL_PATH)
    print(f"✅ Loaded trained model from '{SAVED_MODEL_PATH}'.")

    # Make sure our requested features + target are in the data
    used_features = [col for col in FEATURES if col in val_df.columns]
    if not used_features:
        raise ValueError("None of the requested FEATURES found in validation data.")
    if TARGET not in val_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in validation data.")

    print("\nUsing these features:", used_features)
    print(f"Using target: {TARGET}")

    # Reorder so features come first, then target
    all_cols_ordered = used_features + [TARGET]
    val_df = val_df[all_cols_ordered].copy()

    # Fill or drop any leftover NaNs if needed
    val_df.ffill(inplace=True)

    # Transform using the same scaler
    val_data_scaled = scaler.transform(val_df)  # shape: (rows, len(used_features)+1)

    # Create sequences
    num_features = len(used_features)
    target_idx = num_features  # last column in the scaled data
    X_val, y_val = create_sequences(val_data_scaled, SEQ_LENGTH, num_features, target_idx)
    print(f"\nValidation sequences: X_val.shape={X_val.shape}, y_val.shape={y_val.shape}")

    if X_val.shape[0] == 0:
        print("⚠️ No valid sequences created. Possibly not enough data or too many NaNs.")
        return

    ##########################################
    # 3) MAKE PREDICTIONS & EVALUATE
    ##########################################
    predictions = model.predict(X_val)
    predictions = predictions.flatten()  # shape (samples,)

    # If the target was scaled, we may want to invert scale to original units:
    # We'll do so by creating an array with shape (samples, len(used_features)+1) and placing predictions in the last column,
    # then calling scaler.inverse_transform. The same for y_val.
    # However, if it's fine to compare in scaled space, skip the inverse. 
    # Let's do the inverse so we get actual units:
    # --------------------------------------------------------------
    # Rebuild array for predictions
    val_shape = (len(predictions), len(used_features)+1)
    val_scaled_preds = np.zeros(val_shape, dtype=np.float32)
    val_scaled_true = np.zeros(val_shape, dtype=np.float32)

    # For each sample, the features can be anything, but we just need to place the predicted target in the last column
    # The simplest approach is to fill the features with zeros (or with the last window's features)
    # We'll do zeros for clarity:
    # Then put predictions in the last col
    val_scaled_preds[:, :-1] = 0
    val_scaled_preds[:, -1] = predictions

    val_scaled_true[:, :-1] = 0
    val_scaled_true[:, -1] = y_val

    # Inverse transform
    val_inversed_preds = scaler.inverse_transform(val_scaled_preds)[:, -1]
    val_inversed_true = scaler.inverse_transform(val_scaled_true)[:, -1]

    # Evaluate in real space
    mse = mean_squared_error(val_inversed_true, val_inversed_preds)
    rmse = sqrt(mse)
    mae = mean_absolute_error(val_inversed_true, val_inversed_preds)
    r2 = r2_score(val_inversed_true, val_inversed_preds)

    print("\n================ VALIDATION METRICS ================")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"R^2  = {r2:.4f}")
    print("====================================================")

    ##########################################
    # 4) PLOT RESULTS
    ##########################################
    # We'll show 3 distinct plots:
    #   1) A line plot of predicted vs. actual (subset or full)
    #   2) A scatter plot predicted vs. actual
    #   3) A histogram of residuals

    # 4.1) Line Plot: We'll just plot the last 500 points or so, to avoid an overly long figure
    # Adjust the slice if you want more or less
    n_points_to_plot = min(500, len(val_inversed_true))
    plt.figure()
    plt.plot(val_inversed_true[-n_points_to_plot:], label="Actual")
    plt.plot(val_inversed_preds[-n_points_to_plot:], label="Predicted")
    plt.title("Validation Data: Actual vs. Predicted (Last 500 Points)")
    plt.xlabel("Time Step Index")
    plt.ylabel("Target Temperature (approx)")
    plt.legend()
    plt.show()

    # 4.2) Scatter Plot: Predicted vs. Actual
    plt.figure()
    plt.scatter(val_inversed_true, val_inversed_preds, alpha=0.5)
    plt.title("Validation Data: Predicted vs. Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

    # 4.3) Histogram of Residuals
    residuals = val_inversed_true - val_inversed_preds
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title("Validation Data: Residuals Distribution")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Count")
    plt.show()

    # Done
    print("\n✅ Validation script complete. Plots displayed.")

if __name__ == "__main__":
    main()
