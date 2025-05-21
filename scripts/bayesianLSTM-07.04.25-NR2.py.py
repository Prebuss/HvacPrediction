import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================
# DISABLE GPU to avoid CuDNN error
# ================================
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # This forces CPU usage, ignoring GPU.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import keras_tuner as kt

# For performance metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

###########################################
# 1) CONFIGURATION
###########################################
DATA_FOLDER = "/home/preben/Documents/Master/"
ADAX_FOLDER = os.path.join(DATA_FOLDER, "adax")
VENT_FOLDER = os.path.join(DATA_FOLDER, "vent")
QLARM_FOLDER= os.path.join(DATA_FOLDER, "qlarm")
YR_FOLDER   = os.path.join(DATA_FOLDER, "yr")

OUTPUT_CSV  = "merged_room_temp_dataset.csv"

FEATURES = [
    "energy_consumption", 
    "supplyAirflow", 
    "extractAirflow", 
    "air_temperature", 
    "cloud_area_fraction", 
    "dew_point_temperature", 
    "relative_humidity", 
    "probability_of_precipitation"
]
TARGET = "current_temp"

SEQ_LENGTH = 24
TEST_RATIO = 0.2

# Directory for Keras Tuner
TUNER_DIR = "room_temp_bayes_dir"
PROJECT_NAME = "room_temp_bayes"

###########################################
# 2) MERGE DATA
###########################################
def load_json_files(folder_path):
    data_list = []
    if not os.path.isdir(folder_path):
        print(f"⚠ Folder not found: {folder_path}")
        return data_list
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                data_list.append(json.load(f))
    return data_list

def merge_adax():
    dfs = []
    adax_data = load_json_files(ADAX_FOLDER)
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
        adax_df = pd.concat(dfs, ignore_index=True)
        adax_df.sort_values("timestamp", inplace=True)
        adax_df.reset_index(drop=True, inplace=True)
        return adax_df
    return pd.DataFrame()

def merge_vent():
    vent_data = load_json_files(VENT_FOLDER)
    vent_dfs = []
    for data in vent_data:
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            vent_dfs.append(df)
    if vent_dfs:
        vent_df = pd.concat(vent_dfs, ignore_index=True)
        vent_df.sort_values("timestamp", inplace=True)
        vent_df.reset_index(drop=True, inplace=True)
        return vent_df
    return pd.DataFrame()

def merge_yr():
    yr_dfs = []
    if os.path.isdir(YR_FOLDER):
        for file in sorted(os.listdir(YR_FOLDER)):
            if file.endswith(".json"):
                file_path = os.path.join(YR_FOLDER, file)
                date_str = file[2:12]  # e.g. "yr2025-03-09.json" -> "2025-03-09"
                with open(file_path, "r") as f:
                    data = json.load(f)
                df = pd.DataFrame([
                    {"timestamp": f"{date_str} {entry['time']}", **entry["data"]}
                    for entry in data
                ])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                yr_dfs.append(df)
    if yr_dfs:
        yr_df = pd.concat(yr_dfs, ignore_index=True)
        yr_df.sort_values("timestamp", inplace=True)
        yr_df.reset_index(drop=True, inplace=True)
        return yr_df
    return pd.DataFrame()

def merge_qlarm():
    qlarm_dfs = []
    if not os.path.isdir(QLARM_FOLDER):
        return pd.DataFrame()
    for file in sorted(os.listdir(QLARM_FOLDER)):
        if file.endswith(".csv"):
            file_path = os.path.join(QLARM_FOLDER, file)
            df = pd.read_csv(file_path)
            if "Time" in df.columns:
                df.rename(columns={"Time": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.pivot_table(
                index="timestamp",
                columns="SensorName",
                values="Value",
                aggfunc="mean"
            ).reset_index()
            qlarm_dfs.append(df)
    if qlarm_dfs:
        qlarm_df = pd.concat(qlarm_dfs, ignore_index=True)
        qlarm_df.sort_values("timestamp", inplace=True)
        qlarm_df.reset_index(drop=True, inplace=True)
        return qlarm_df
    return pd.DataFrame()

def merge_all_data():
    adax_df = merge_adax()
    vent_df = merge_vent()
    yr_df   = merge_yr()
    qlarm_df= merge_qlarm()

    merged_df = (
        adax_df.merge(vent_df, on="timestamp", how="outer")
               .merge(yr_df, on="timestamp", how="outer")
               .merge(qlarm_df, on="timestamp", how="outer")
    )
    merged_df.sort_values("timestamp", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    merged_df.ffill(inplace=True)

    if OUTPUT_CSV:
        merged_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved merged data to {OUTPUT_CSV} with shape={merged_df.shape}")
    return merged_df

###########################################
# 3) CREATE SEQUENCES
###########################################
def create_sequences(dataX, dataY, seq_length):
    X, y = [], []
    for i in range(len(dataX) - seq_length):
        seq_x = dataX[i : i + seq_length]
        seq_y = dataY[i + seq_length]
        if not np.isnan(seq_x).any() and not np.isnan(seq_y):
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

###########################################
# 4) LSTM MODEL
###########################################
def build_lstm_model(hp):
    model = Sequential()

    # number of LSTM layers
    num_lstm_layers = hp.Int('num_lstm_layers', 1, 2, default=1)

    units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32, default=64)
    dropout_1 = hp.Float('dropout_1', 0.0, 0.5, step=0.1, default=0.2)
    rec_dropout_1 = hp.Float('rec_dropout_1', 0.0, 0.5, step=0.1, default=0.2)

    input_shape = (SEQ_LENGTH, len(FEATURES))

    if num_lstm_layers > 1:
        model.add(
            LSTM(
                units_1,
                return_sequences=True,
                input_shape=input_shape,
                dropout=dropout_1,
                recurrent_dropout=rec_dropout_1
            )
        )
        units_2 = hp.Int('units_2', min_value=32, max_value=128, step=16, default=64)
        dropout_2 = hp.Float('dropout_2', 0.0, 0.5, step=0.1, default=0.2)
        rec_dropout_2 = hp.Float('rec_dropout_2', 0.0, 0.5, step=0.1, default=0.2)

        model.add(
            LSTM(
                units_2,
                dropout=dropout_2,
                recurrent_dropout=rec_dropout_2
            )
        )
    else:
        # single layer
        model.add(
            LSTM(
                units_1,
                input_shape=input_shape,
                dropout=dropout_1,
                recurrent_dropout=rec_dropout_1
            )
        )

    dense_units = hp.Int('dense_units', 16, 64, step=16, default=32)
    l2_reg = hp.Float('l2_reg', 1e-4, 1e-2, sampling='log', default=1e-3)

    model.add(Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dense(1))

    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='mse'
    )
    return model

###########################################
# 5) EVALUATION
###########################################
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test).flatten()
    mse  = mean_squared_error(y_test, preds)
    rmse = sqrt(mse)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print("\n===== TEST SET METRICS =====")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"R^2  = {r2:.4f}")

    # Plots
    # 1) line plot (last 500 points)
    n_points = min(500, len(y_test))
    plt.figure()
    plt.plot(y_test[-n_points:], label="Actual")
    plt.plot(preds[-n_points:], label="Predicted")
    plt.title("Test Set: Actual vs. Predicted (Last 500 points)")
    plt.xlabel("Step")
    plt.ylabel("Scaled Temp")
    plt.legend()
    plt.show()

    # 2) scatter
    plt.figure()
    plt.scatter(y_test, preds, alpha=0.3)
    plt.title("Test Set: Predicted vs. Actual")
    plt.xlabel("Actual (scaled)")
    plt.ylabel("Predicted (scaled)")
    plt.show()

    # 3) residual histogram
    residuals = y_test - preds
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution (Test Set)")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Count")
    plt.show()

###########################################
# 6) PLOT HYPERPARAMETERS
###########################################
def plot_hyperparams(tuner):
    """
    Plots each trial's final val_loss vs. its hyperparameters,
    giving a sense of which combos worked best.
    """
    trials = tuner.oracle.get_best_trials()
    # But we want all trials, not just the best
    all_trials = tuner.oracle.trials.values()

    hp_keys = list(all_trials)[0].hyperparameters.values.keys() if all_trials else []
    # We'll create subplots for each hyperparameter
    # and plot final val_loss vs. that hyperparameter.
    n_keys = len(hp_keys)
    plt.figure(figsize=(6 * n_keys, 4))

    for i, hp_name in enumerate(hp_keys, 1):
        plt.subplot(1, n_keys, i)
        # Extract hp_name and final val_loss from each trial
        hp_vals = []
        losses = []
        for t in all_trials:
            val = t.hyperparameters.values.get(hp_name, None)
            loss = t.score  # or t.metrics.get("val_loss") etc.
            if loss is not None and val is not None:
                hp_vals.append(val)
                losses.append(loss)

        plt.scatter(hp_vals, losses)
        plt.title(f"{hp_name} vs val_loss")
        plt.xlabel(hp_name)
        plt.ylabel("val_loss")
    plt.tight_layout()
    plt.show()

###########################################
# MAIN
###########################################
def main():
    # Merge Data
    merged_df = merge_all_data()

    # Check columns
    for c in [TARGET] + FEATURES:
        if c not in merged_df.columns:
            raise ValueError(f"Column '{c}' not found in data.")

    # Prepare data
    df_used = merged_df[FEATURES + [TARGET]].copy()

    # Scale
    scaler = MinMaxScaler()
    df_used[FEATURES + [TARGET]] = scaler.fit_transform(df_used[FEATURES + [TARGET]])

    # Create sequences
    data_array   = df_used[FEATURES].values
    target_array = df_used[TARGET].values
    X, y = create_sequences(data_array, target_array, SEQ_LENGTH)
    print(f"Sequences created:\nX.shape = {X.shape}, y.shape = {y.shape}")

    # Time-based split
    n = len(X)
    split_idx = int(n * (1 - TEST_RATIO))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test,  y_test  = X[split_idx:], y[split_idx:]
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    # Tuner
    tuner = kt.BayesianOptimization(
        build_lstm_model,
        objective="val_loss",
        max_trials=10,
        num_initial_points=3,
        alpha=1e-4,
        beta=2.6,
        directory=TUNER_DIR,
        project_name=PROJECT_NAME,
        overwrite=True  # If we want to re-run from scratch
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

    print("\nStarting Bayesian optimization search...")
    try:
        tuner.search(
            X_train,
            y_train,
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
    except Exception as e:
        print("Tuner search failed with exception:", e)

    # Plot hyperparams
    plot_hyperparams(tuner)

    # Get best model
    best_models = tuner.get_best_models(num_models=1)
    if not best_models:
        print("No best model found. Tuner might have failed due to CuDNN issues.")
        return

    best_model = best_models[0]
    # Evaluate on test
    evaluate_model(best_model, X_test, y_test)

    best_model.save("room_temp_best_bayes_cpu.keras")
    print("Best model saved as 'room_temp_best_bayes_cpu.keras'.")


if __name__ == "__main__":
    main()
