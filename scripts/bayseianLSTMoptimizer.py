import os
# Force TensorFlow to use standard float types (disable ml_dtypes)
os.environ["TF_USE_ML_DTYPES"] = "0"

import json
import shutil
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
import pickle

# =========================================
# 1) DATA MERGING SECTION
# =========================================

# Adjust these if needed
DATA_FOLDER = "/home/preben/Documents/Master/"
YR_FOLDER = os.path.join(DATA_FOLDER, "yr")
VENT_FOLDER = os.path.join(DATA_FOLDER, "vent")
ADAX_FOLDER = os.path.join(DATA_FOLDER, "adax")
QLARM_FOLDER = os.path.join(DATA_FOLDER, "qlarm")

OUTPUT_CSV = "merged_hvac_dataset.csv"

# -----------------------------------------
# Helper to load JSON files from a folder
# -----------------------------------------
def load_json_files(folder_path):
    data_list = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                data_list.append(json.load(f))
    return data_list

# -----------------------------------------
# Merge YR data
# -----------------------------------------
def merge_yr_data():
    yr_dfs = []
    if not os.path.isdir(YR_FOLDER):
        print(f"⚠️  YR folder not found: {YR_FOLDER}")
        return pd.DataFrame()
    
    for file in sorted(os.listdir(YR_FOLDER)):
        if file.endswith(".json"):
            file_path = os.path.join(YR_FOLDER, file)
            # e.g. "yr2025-03-09.json" -> date_str = "2025-03-09"
            date_str = file[2:12]
            with open(file_path, "r") as f:
                data = json.load(f)
            # Each entry has a "time" and "data" dict
            df = pd.DataFrame([
                {
                    "timestamp": f"{date_str} {entry['time']}",
                    **entry["data"]
                }
                for entry in data
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            yr_dfs.append(df)

    if yr_dfs:
        yr_df = pd.concat(yr_dfs, ignore_index=True)
        yr_df.sort_values(by="timestamp", inplace=True)
        yr_df.reset_index(drop=True, inplace=True)
        return yr_df
    else:
        return pd.DataFrame()

# -----------------------------------------
# Merge Vent data
# -----------------------------------------
def merge_vent_data():
    vent_dfs = []
    if not os.path.isdir(VENT_FOLDER):
        print(f"⚠️  Vent folder not found: {VENT_FOLDER}")
        return pd.DataFrame()
    
    vent_data = load_json_files(VENT_FOLDER)
    for data in vent_data:
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            vent_dfs.append(df)
    if vent_dfs:
        vent_df = pd.concat(vent_dfs, ignore_index=True)
        vent_df.sort_values(by="timestamp", inplace=True)
        vent_df.reset_index(drop=True, inplace=True)
        return vent_df
    else:
        return pd.DataFrame()

# -----------------------------------------
# Merge Adax data
# -----------------------------------------
def merge_adax_data():
    adax_dfs = []
    if not os.path.isdir(ADAX_FOLDER):
        print(f"⚠️  Adax folder not found: {ADAX_FOLDER}")
        return pd.DataFrame()
    
    adax_data = load_json_files(ADAX_FOLDER)
    for data in adax_data:
        temp_list = []
        for entry in data:
            timestamp = entry.get("timestamp")
            rooms = entry.get("rooms", [])
            for room in rooms:
                energy_sum = sum(dev["energy"] for dev in room.get("devices", []))
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
            adax_dfs.append(df)
    if adax_dfs:
        adax_df = pd.concat(adax_dfs, ignore_index=True)
        adax_df.sort_values(by="timestamp", inplace=True)
        adax_df.reset_index(drop=True, inplace=True)
        return adax_df
    else:
        return pd.DataFrame()

# -----------------------------------------
# Merge Qlarm data
# -----------------------------------------
def merge_qlarm_data():
    qlarm_dfs = []
    if not os.path.isdir(QLARM_FOLDER):
        print(f"⚠️  Qlarm folder not found: {QLARM_FOLDER}")
        return pd.DataFrame()
    
    for file in os.listdir(QLARM_FOLDER):
        if file.endswith(".csv"):
            file_path = os.path.join(QLARM_FOLDER, file)
            df = pd.read_csv(file_path)
            # rename if needed
            if "Time" in df.columns:
                df.rename(columns={"Time": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # pivot data if needed
            df = df.pivot_table(
                index="timestamp",
                columns="SensorName",
                values="Value",
                aggfunc="mean"
            ).reset_index()
            qlarm_dfs.append(df)

    if qlarm_dfs:
        qlarm_df = pd.concat(qlarm_dfs, ignore_index=True)
        qlarm_df.sort_values(by="timestamp", inplace=True)
        qlarm_df.reset_index(drop=True, inplace=True)
        return qlarm_df
    else:
        return pd.DataFrame()

def merge_all_data_and_save():
    """Merges data from YR, Vent, Adax, Qlarm into a single DataFrame,
    computes 'Preben - Office Average - Temperature' from Qlarm temperature columns,
    and saves to OUTPUT_CSV. Returns the merged DataFrame."""
    print("🔄 Loading data from multiple files...")

    yr_df = merge_yr_data()
    vent_df = merge_vent_data()
    adax_df = merge_adax_data()
    qlarm_df = merge_qlarm_data()

    print(f"YR DataFrame shape: {yr_df.shape}")
    print(f"Vent DataFrame shape: {vent_df.shape}")
    print(f"Adax DataFrame shape: {adax_df.shape}")
    print(f"Qlarm DataFrame shape: {qlarm_df.shape}")

    # Merge on 'timestamp'
    merged_df = (
        adax_df.merge(vent_df, on="timestamp", how="outer")
               .merge(yr_df, on="timestamp", how="outer")
               .merge(qlarm_df, on="timestamp", how="outer")
    )
    merged_df.sort_values("timestamp", inplace=True)
    merged_df.ffill(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    print(f"Merged dataset shape: {merged_df.shape}")

    # Compute "Preben - Office Average - Temperature"
    qlarm_temperature_cols = [
        col for col in merged_df.columns
        if "Temperature" in col and col != "Preben - Office Average - Temperature"
    ]
    if qlarm_temperature_cols:
        merged_df["Preben - Office Average - Temperature"] = merged_df[qlarm_temperature_cols].mean(axis=1)
        print(f"Computed 'Preben - Office Average - Temperature' using {len(qlarm_temperature_cols)} Qlarm temperature columns.")
    else:
        print("⚠️ No Qlarm temperature columns found. 'Preben - Office Average - Temperature' not computed.")

    # Save CSV
    output_path = os.path.join(DATA_FOLDER, OUTPUT_CSV)
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Merged dataset saved to '{output_path}' with {len(merged_df)} entries.")

    return merged_df

# =========================================
# 2) MODEL TRAINING WITH BAYESIAN OPTIMIZATION
# =========================================

def main():
    # Merge data & save CSV
    merged_df = merge_all_data_and_save()

    # ========== SELECT FEATURES & TARGET ==========
    requested_features = [
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
    target = 'target_temp'

    # Keep only columns that exist
    available_features = [col for col in requested_features if col in merged_df.columns]
    if not available_features:
        raise ValueError("None of the requested features are present in merged data.")
    if target not in merged_df.columns:
        raise ValueError(f"Target column '{target}' not found in merged data.\n"
                         f"Columns available: {list(merged_df.columns)}")

    features = available_features
    print("\nFeatures requested:", requested_features)
    print("Features found in data:", features)
    print(f"Target used for prediction: {target}")

    # ========== PREPARE DATA ==========
    scaler = MinMaxScaler()
    cols_to_scale = features + [target]
    merged_df[cols_to_scale] = scaler.fit_transform(merged_df[cols_to_scale])

    # Save the scaler for later use
    with open("min_max_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Sequence length (24 timesteps)
    SEQ_LENGTH = 24

    # Create sequences
    def create_sequences(data, target_column, seq_length):
        X, y = [], []
        target_idx = cols_to_scale.index(target_column)
        num_features = len(features)

        for i in range(len(data) - seq_length):
            seq_X = data[i:i+seq_length, :num_features]
            seq_y = data[i + seq_length, target_idx]
            if not np.isnan(seq_X).any() and not np.isnan(seq_y):
                X.append(seq_X)
                y.append(seq_y)
        return np.array(X), np.array(y)

    # Reorder columns so that features come first and target is last
    all_columns_ordered = features + [target]
    merged_data_array = merged_df[all_columns_ordered].values

    X, y = create_sequences(merged_data_array, target, SEQ_LENGTH)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    print(f"\nSequence creation complete: X.shape={X.shape}, y.shape={y.shape}")

    # Train-test split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("y_train:", y_train.shape, "y_test:", y_test.shape)
    print("Any NaNs in X_train?", np.isnan(X_train).any())
    print("Any NaNs in y_train?", np.isnan(y_train).any())

    # ========== BUILD MODEL + BAYESIAN OPTIMIZATION ==========
    def build_model(hp):
        model = Sequential()
        
        # Number of LSTM layers
        num_lstm_layers = hp.Int('num_lstm_layers', 1, 2, default=2)
        input_shape = (SEQ_LENGTH, len(features))
        
        # First LSTM layer
        units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32, default=128)
        dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
        rec_dropout_1 = hp.Float('rec_dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
        
        if num_lstm_layers > 1:
            model.add(LSTM(
                units_1,
                return_sequences=True,
                input_shape=input_shape,
                dropout=dropout_1,
                recurrent_dropout=rec_dropout_1
            ))
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
                input_shape=input_shape,
                dropout=dropout_1,
                recurrent_dropout=rec_dropout_1
            ))

        dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16, default=32)
        l2_reg = hp.Float('l2_reg', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
        model.add(Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(Dense(1))  # Single output layer
        
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss='mse'
        )
        return model

    # Set overwrite=True to ensure a fresh tuner run
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=20,
        num_initial_points=5,
        alpha=1e-4,
        beta=2.6,
        directory='hvac_tuner_bayes_dir',
        project_name='lstm_hvac_tuning_bayes',
        overwrite=True
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )

    print("\n🔍 Starting BAYESIAN hyperparameter search...")
    tuner.search(
        X_train,
        y_train,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n✅ Best hyperparameters found (Bayesian):")
    print(best_hp.values)

    test_loss = best_model.evaluate(X_test, y_test)
    print(f"📉 Test Loss (MSE): {test_loss}")

    best_model.save("lstm_hvac_forecast_best_bayes.keras")
    print("✅ Best Bayesian-optimized model saved as 'lstm_hvac_forecast_best_bayes.keras'.")


if __name__ == "__main__":
    main()
