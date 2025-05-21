import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras_tuner as kt

###########################################
# 1) CONFIGURATION
###########################################

DATA_FOLDER = "/home/preben/Documents/Master/"

ADAX_FOLDER = os.path.join(DATA_FOLDER, "adax")
VENT_FOLDER = os.path.join(DATA_FOLDER, "vent")
QLARM_FOLDER= os.path.join(DATA_FOLDER, "qlarm")
YR_FOLDER   = os.path.join(DATA_FOLDER, "yr")

# The final CSV after merging (optional)
OUTPUT_CSV  = "merged_room_temp_dataset.csv"

# Features from your correlation heat map
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

SEQ_LENGTH = 24  # Number of time steps in each training sequence

###########################################
# 2) MERGING DATA
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
    """Merge Adax JSON data from ADAX_FOLDER."""
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
    """Merge Vent JSON data from VENT_FOLDER."""
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
    """Merge YR JSON data from YR_FOLDER."""
    yr_dfs = []
    if os.path.isdir(YR_FOLDER):
        for file in sorted(os.listdir(YR_FOLDER)):
            if file.endswith(".json"):
                file_path = os.path.join(YR_FOLDER, file)
                # e.g. "yr2025-03-09.json" -> date_str = "2025-03-09"
                date_str = file[2:12]
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
    """Merge Qlarm CSV from QLARM_FOLDER."""
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
            # pivot sensors
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
    """Merge Adax, Vent, YR, Qlarm into one DataFrame on 'timestamp'."""
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

    # Forward-fill missing data
    merged_df.ffill(inplace=True)

    # (Optional) if you want to average all Qlarm temperature columns, do so here
    # or drop columns not in your selected features, etc.

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
# 4) LSTM MODEL WITH BAYESIAN OPTIMIZATION
###########################################
def build_lstm_model(hp):
    model = Sequential()
    
    # Choose 1 or 2 layers
    num_lstm_layers = hp.Int('num_lstm_layers', 1, 2, default=1)

    # 1st LSTM layer
    units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32, default=64)
    dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
    rec_dropout_1 = hp.Float('rec_dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=0.2)

    input_shape = (SEQ_LENGTH, len(FEATURES))

    if num_lstm_layers > 1:
        # First LSTM with return_sequences=True
        model.add(
            LSTM(
                units_1,
                return_sequences=True,
                input_shape=input_shape,
                dropout=dropout_1,
                recurrent_dropout=rec_dropout_1
            )
        )
        
        # 2nd LSTM
        units_2 = hp.Int('units_2', min_value=32, max_value=128, step=16, default=64)
        dropout_2 = hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
        rec_dropout_2 = hp.Float('rec_dropout_2', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
        
        model.add(
            LSTM(
                units_2,
                dropout=dropout_2,
                recurrent_dropout=rec_dropout_2
            )
        )
    else:
        # Just one layer
        model.add(
            LSTM(
                units_1,
                input_shape=input_shape,
                dropout=dropout_1,
                recurrent_dropout=rec_dropout_1
            )
        )

    # Dense layer
    dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16, default=32)
    l2_reg = hp.Float('l2_reg', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)

    model.add(
        Dense(
            dense_units, 
            activation="relu", 
            kernel_regularizer=regularizers.l2(l2_reg)
        )
    )
    model.add(Dense(1))  # Predict the room temperature

    # Learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='mse'
    )
    return model

###########################################
# 5) MAIN
###########################################
def main():
    # 1) Merge data
    merged_df = merge_all_data()

    # 2) Check columns
    for col in [TARGET] + FEATURES:
        if col not in merged_df.columns:
            raise ValueError(f"Column '{col}' not found in merged data.")

    # 3) Select only the columns we need
    df_used = merged_df[FEATURES + [TARGET]].copy()

    # 4) Scale
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_used[FEATURES + [TARGET]] = scaler.fit_transform(df_used[FEATURES + [TARGET]])

    # 5) Create sequences
    data_array = df_used[FEATURES].values
    target_array = df_used[TARGET].values

    X, y = create_sequences(data_array, target_array, SEQ_LENGTH)
    print("Sequences created:")
    print(f"X.shape = {X.shape}, y.shape = {y.shape}")

    # 6) Train-test split (time-series style: no shuffle)
    #    We'll do something simple: 80% train, 20% test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    # 7) Bayesian optimization
    tuner = kt.BayesianOptimization(
        build_lstm_model,
        objective="val_loss",
        max_trials=10,
        num_initial_points=3,
        alpha=1e-4,
        beta=2.6,
        directory="room_temp_bayes_dir",
        project_name="room_temp_bayes"
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

    print("\nStarting Bayesian optimization search...")
    tuner.search(
        X_train, y_train,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\nBest hyperparameters found:")
    print(best_hp.values)

    # Evaluate final model
    test_loss = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Final test MSE: {test_loss}")

    # Save best model
    best_model.save("room_temp_best_bayes.keras")
    print("Best model saved as 'room_temp_best_bayes.keras'.")

if __name__ == "__main__":
    main()
