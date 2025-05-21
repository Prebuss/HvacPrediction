import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras_tuner as kt

from tqdm import tqdm  # for progress bar

# ========== 1) SETUP & DATA LOADING ========== #

# Ensure TensorFlow Uses GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ TensorFlow is using GPU 🚀")
    except RuntimeError as e:
        print(f"⚠️ GPU Configuration Error: {e}")
else:
    print("⚠️ No GPU detected. Running on CPU.")

DATA_FOLDER = "/home/preben/Documents/Master/"  # Adjust if needed

def load_json_files(folder_path):
    """Generic function to load all JSON files in a folder."""
    data_list = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                data_list.append(json.load(f))
    return data_list

print("🔄 Loading data from multiple files...")

# --- YR Weather Forecast Data ---
yr_dfs = []
yr_folder = os.path.join(DATA_FOLDER, "yr")
if os.path.isdir(yr_folder):
    for file in sorted(os.listdir(yr_folder)):
        if file.endswith(".json"):
            file_path = os.path.join(yr_folder, file)
            date_str = file[2:12]  # e.g. "yr2025-03-09.json" => "2025-03-09"
            with open(file_path, "r") as f:
                data = json.load(f)
            df = pd.DataFrame([
                {"timestamp": f"{date_str} {entry['time']}", **entry["data"]}
                for entry in data
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            yr_dfs.append(df)
yr_df = pd.concat(yr_dfs, ignore_index=True) if yr_dfs else pd.DataFrame()

# --- Ventilation Data ---
vent_dfs = []
vent_folder = os.path.join(DATA_FOLDER, "vent")
if os.path.isdir(vent_folder):
    vent_data = load_json_files(vent_folder)
    for data in vent_data:
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        vent_dfs.append(df)
vent_df = pd.concat(vent_dfs, ignore_index=True) if vent_dfs else pd.DataFrame()

# --- Adax Heating Panel Data ---
adax_dfs = []
adax_folder = os.path.join(DATA_FOLDER, "adax")
if os.path.isdir(adax_folder):
    adax_data = load_json_files(adax_folder)
    for data in adax_data:
        temp_list = []
        for entry in data:
            timestamp = entry["timestamp"]
            for room in entry["rooms"]:
                temp_list.append({
                    "timestamp": pd.to_datetime(timestamp),
                    "room_id": room["id"],
                    "room_name": room["name"],
                    "target_temp": room["targetTemperature"],
                    "current_temp": room["currentTemperature"],
                    "energy_consumption": sum(dev["energy"] for dev in room["devices"])
                })
        df = pd.DataFrame(temp_list)
        adax_dfs.append(df)
adax_df = pd.concat(adax_dfs, ignore_index=True) if adax_dfs else pd.DataFrame()

# --- Qlarm CSV Data ---
qlarm_dfs = []
qlarm_folder = os.path.join(DATA_FOLDER, "qlarm")
if os.path.isdir(qlarm_folder):
    for file in os.listdir(qlarm_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(qlarm_folder, file)
            df = pd.read_csv(file_path)
            df.rename(columns={"Time": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.pivot_table(
                index="timestamp",
                columns="SensorName",
                values="Value",
                aggfunc="mean"
            ).reset_index()
            qlarm_dfs.append(df)
qlarm_df = pd.concat(qlarm_dfs, ignore_index=True) if qlarm_dfs else pd.DataFrame()

# ========== 2) MERGE ALL ========== #
merged_df = (
    adax_df.merge(vent_df, on="timestamp", how="outer")
           .merge(yr_df, on="timestamp", how="outer")
           .merge(qlarm_df, on="timestamp", how="outer")
)

merged_df.sort_values("timestamp", inplace=True)
merged_df.ffill(inplace=True)  # Forward-fill missing

merged_df.to_csv("merged_hvac_dataset.csv", index=False)
print(f"✅ Merged dataset saved with {len(merged_df)} entries.")
print("Columns in merged_df:", merged_df.columns.tolist())
print("NaNs in target_temp:", merged_df["target_temp"].isna().sum())

# ========== 3) SELECT FEATURES & TARGET ========== #
target = "target_temp"

# Potential input features. Excluding "target_temp" itself because it's our label.
all_potential_features = [
    "current_temp",
    "energy_consumption",
    "co2", "supplyAirflow", "extractAirflow", "supplyAirDuctPressure",
    "extractAirDuctPressure", "supplyAirFanSpeedLevel", "extractAirFanSpeedLevel",
    "supplyAirTemperature", "extractAirTemperature", "outdoorTemperature",
    "reheatLevel", "coolingLevel", "heatExchangerRegulator", "RhxEfficiency",
    "air_temperature", "relative_humidity", "wind_speed"
]

# Keep only those that exist
all_potential_features = [f for f in all_potential_features if f in merged_df.columns]

# ========== 4) SCALING ========== #
# We'll scale the input features AND the target column.
scaler_features = all_potential_features + [target]
scaler_features = list(dict.fromkeys(scaler_features))  # remove duplicates
scaler_features = [col for col in scaler_features if col in merged_df.columns]

scaler = MinMaxScaler()
merged_df[scaler_features] = scaler.fit_transform(merged_df[scaler_features])

# ========== 5) CREATE SEQUENCES (FEATURES vs. TARGET) ========== #
SEQ_LENGTH = 24

def create_sequences(df, feature_cols, target_col, seq_len=24):
    """
    Build sequences from separate input features vs. target arrays:
        - X shape => (num_samples, seq_len, len(feature_cols))
        - y shape => (num_samples,)
    """
    data_x = df[feature_cols].values  # shape (N, #features)
    data_y = df[target_col].values    # shape (N,)

    X, y = [], []
    for i in range(len(df) - seq_len):
        seq_features = data_x[i : i+seq_len]  # shape (seq_len, #features)
        seq_target = data_y[i+seq_len]        # single value
        if not np.isnan(seq_features).any() and not np.isnan(seq_target):
            X.append(seq_features)
            y.append(seq_target)

    return np.array(X), np.array(y)

# ========== 6) BUILD MODEL (LSTM) FOR KERAS TUNER ========== #
def build_model(hp, seq_length, n_features):
    model = Sequential()
    
    # Number of LSTM layers
    num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=2, default=2)
    
    # First LSTM layer
    units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32, default=128)
    dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
    rec_dropout_1 = hp.Float('rec_dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
    
    if num_lstm_layers > 1:
        model.add(LSTM(
            units_1,
            return_sequences=True,
            input_shape=(seq_length, n_features),
            dropout=dropout_1,
            recurrent_dropout=rec_dropout_1
        ))
        # Second LSTM layer
        units_2 = hp.Int('units_2', min_value=32, max_value=128, step=16, default=64)
        dropout_2 = hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
        rec_dropout_2 = hp.Float('rec_dropout_2', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
        model.add(LSTM(
            units_2,
            dropout=dropout_2,
            recurrent_dropout=rec_dropout_2
        ))
    else:
        model.add(LSTM(
            units_1,
            input_shape=(seq_length, n_features),
            dropout=dropout_1,
            recurrent_dropout=rec_dropout_1
        ))
    
    # Dense layer
    dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16, default=32)
    l2_reg = hp.Float('l2_reg', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
    model.add(Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)))
    
    # Final output
    model.add(Dense(1))
    
    # Compile
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model

# ========== 7) TRAIN & EVALUATE ON ONE FEATURE SUBSET ========== #
def train_and_evaluate_feature_subset(subset_feats):
    # Build sequences for just these subset_feats as X, plus 'target_temp' as y
    X, y = create_sequences(merged_df, subset_feats, target, seq_len=SEQ_LENGTH)

    # If there's no data or too small
    if X.shape[0] == 0:
        raise ValueError("Not enough valid sequences for this feature subset.")
    
    # Train-test split (no shuffle, time-series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Keras Tuner
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, SEQ_LENGTH, len(subset_feats)),
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='hvac_tuner_dir',
        project_name=f'lstm_hvac_tuning_{len(subset_feats)}f',
        overwrite=True  # Start fresh each time
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6
    )
    
    tuner.search(
        X_train, y_train,
        epochs=10,  # short for example
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    best_model = tuner.get_best_models(num_models=1)[0]
    test_loss = best_model.evaluate(X_test, y_test, verbose=0)
    return test_loss

# ========== 8) MAIN LOOP: Incrementally Add Features to Predict target_temp ========== #
results = []
print()

for i in tqdm(range(1, len(all_potential_features) + 1), desc="Feature Subset"):
    subset_features = all_potential_features[:i]
    print(f"\n🚀 Training with subset of size {len(subset_features)} -> {subset_features}")

    try:
        mse = train_and_evaluate_feature_subset(subset_features)
        results.append((len(subset_features), subset_features, mse))
        print(f"   MSE for these {len(subset_features)} features: {mse:.6f}")
    except Exception as e:
        print(f"❌ Error with subset {subset_features}: {e}")
        break

# ========== 9) SUMMARY ========== #
print("\n===== Summary of Results by Feature Subset Size =====")
for size, feats, mse_val in results:
    print(f"{size} features -> MSE: {mse_val:.6f} | Features: {feats}")
