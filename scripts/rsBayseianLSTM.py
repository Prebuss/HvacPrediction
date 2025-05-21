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
import keras_tuner as kt

###########################################
# 1) CONFIGURATION
###########################################

tf.config.set_visible_devices([], 'GPU')

DATA_FOLDER = "/home/preben/Documents/Master/"  

# Folders for TRAIN data ("val" folders)
VAL_ADAX_FOLDER = os.path.join(DATA_FOLDER, "valAdax")
VAL_QLARM_FOLDER = os.path.join(DATA_FOLDER, "valQlarm")
VAL_VENT_FOLDER = os.path.join(DATA_FOLDER, "valVent")
VAL_YR_FOLDER = os.path.join(DATA_FOLDER, "valYr")

# Folders for VALIDATION data ("regular" folders)
ADAX_FOLDER = os.path.join(DATA_FOLDER, "adax")
QLARM_FOLDER = os.path.join(DATA_FOLDER, "qlarm")
VENT_FOLDER = os.path.join(DATA_FOLDER, "vent")
YR_FOLDER   = os.path.join(DATA_FOLDER, "yr")

# Output CSV (optional)
TRAIN_CSV = "merged_val_dataset.csv"
VAL_CSV   = "merged_regular_dataset.csv"

SEQ_LENGTH = 24

# Features & target
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
TARGET = 'target_temp'

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

def merge_adax(folder_path):
    dfs = []
    adax_data = load_json_files(folder_path)
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

def merge_vent(folder_path):
    vent_data = load_json_files(folder_path)
    vent_dfs = []
    for data in vent_data:
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            vent_dfs.append(df)
    if vent_dfs:
        out = pd.concat(vent_dfs, ignore_index=True)
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out
    return pd.DataFrame()

def merge_yr(folder_path):
    yr_dfs = []
    if os.path.isdir(folder_path):
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".json"):
                file_path = os.path.join(folder_path, file)
                # e.g., "yr2025-03-09.json" -> date_str = "2025-03-09"
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
        out = pd.concat(yr_dfs, ignore_index=True)
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out
    return pd.DataFrame()

def merge_qlarm(folder_path):
    qlarm_dfs = []
    if not os.path.isdir(folder_path):
        return pd.DataFrame()
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
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
        out = pd.concat(qlarm_dfs, ignore_index=True)
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out
    return pd.DataFrame()

def merge_data(adax_folder, qlarm_folder, vent_folder, yr_folder, output_csv=None):
    adax_df = merge_adax(adax_folder)
    qlarm_df = merge_qlarm(qlarm_folder)
    vent_df  = merge_vent(vent_folder)
    yr_df    = merge_yr(yr_folder)

    merged_df = (
        adax_df.merge(vent_df, on="timestamp", how="outer")
               .merge(yr_df, on="timestamp", how="outer")
               .merge(qlarm_df, on="timestamp", how="outer")
    )
    merged_df.sort_values("timestamp", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    # Forward fill
    merged_df.ffill(inplace=True)

    # "Preben - Office Average - Temperature"
    qlarm_temperature_cols = [
        c for c in merged_df.columns
        if "Temperature" in c and c != "Preben - Office Average - Temperature"
    ]
    if qlarm_temperature_cols:
        merged_df["Preben - Office Average - Temperature"] = merged_df[qlarm_temperature_cols].mean(axis=1)

    if output_csv:
        merged_df.to_csv(output_csv, index=False)
        print(f"Saved merged data to {output_csv} with shape={merged_df.shape}")

    return merged_df

###########################################
# 3) CREATE SEQUENCES
###########################################
def create_sequences(dataX, dataY, seq_length):
    X, y = [], []
    for i in range(len(dataX) - seq_length):
        seq_x = dataX[i : i + seq_length]
        seq_y = dataY[i + seq_length]  # predict the step after the window
        if not np.isnan(seq_x).any() and not np.isnan(seq_y):
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

###########################################
# 4) BUILD MODEL
###########################################
def build_model(hp, init_params=None):
    model = Sequential()

    if init_params:
        num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=2, default=init_params['num_lstm_layers'])
        units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32, default=init_params['units_1'])
        dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=init_params['dropout_1'])
        rec_dropout_1 = hp.Float('rec_dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=init_params['rec_dropout_1'])
    else:
        num_lstm_layers = hp.Int('num_lstm_layers', 1, 2, default=2)
        units_1 = hp.Int('units_1', 32, 256, step=32, default=128)
        dropout_1 = hp.Float('dropout_1', 0.0, 0.5, step=0.1, default=0.2)
        rec_dropout_1 = hp.Float('rec_dropout_1', 0.0, 0.5, step=0.1, default=0.2)

    input_shape = (SEQ_LENGTH, len(FEATURES))

    model.add(LSTM(
        units_1,
        return_sequences=(num_lstm_layers > 1),
        input_shape=input_shape,
        dropout=dropout_1,
        recurrent_dropout=rec_dropout_1
    ))

    if num_lstm_layers > 1:
        if init_params:
            units_2 = hp.Int('units_2', 32, 128, step=16, default=init_params['units_2'])
            dropout_2 = hp.Float('dropout_2', 0.0, 0.5, step=0.1, default=init_params['dropout_2'])
            rec_dropout_2 = hp.Float('rec_dropout_2', 0.0, 0.5, step=0.1, default=init_params['rec_dropout_2'])
        else:
            units_2 = hp.Int('units_2', 32, 128, step=16, default=64)
            dropout_2 = hp.Float('dropout_2', 0.0, 0.5, step=0.1, default=0.2)
            rec_dropout_2 = hp.Float('rec_dropout_2', 0.0, 0.5, step=0.1, default=0.2)

        model.add(LSTM(
            units_2,
            dropout=dropout_2,
            recurrent_dropout=rec_dropout_2
        ))

    if init_params:
        dense_units = hp.Int('dense_units', 16, 64, step=16, default=init_params['dense_units'])
        l2_reg      = hp.Float('l2_reg', min_value=1e-4, max_value=1e-2, sampling='log', default=init_params['l2_reg'])
        lr          = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=init_params['learning_rate'])
    else:
        dense_units = hp.Int('dense_units', 16, 64, step=16, default=32)
        l2_reg      = hp.Float('l2_reg', 1e-4, 1e-2, sampling='log', default=1e-3)
        lr          = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)

    model.add(Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="mse"
    )
    return model

###########################################
# 5) MAIN
###########################################
def main():
    print("===== MERGING TRAIN (val) DATA =====")
    train_df = merge_data(
        VAL_ADAX_FOLDER, VAL_QLARM_FOLDER, VAL_VENT_FOLDER, VAL_YR_FOLDER,
        output_csv=TRAIN_CSV
    )
    print("===== MERGING VALIDATION DATA =====")
    val_df = merge_data(
        ADAX_FOLDER, QLARM_FOLDER, VENT_FOLDER, YR_FOLDER,
        output_csv=VAL_CSV
    )

    # Ensure target exists
    if TARGET not in train_df.columns:
        raise ValueError(f"Target '{TARGET}' not found in train data.")
    if TARGET not in val_df.columns:
        raise ValueError(f"Target '{TARGET}' not found in val data.")

    # Find common features
    train_features = [f for f in FEATURES if f in train_df.columns]
    val_features   = [f for f in FEATURES if f in val_df.columns]
    common_features = sorted(set(train_features).intersection(val_features))
    if not common_features:
        raise ValueError("No common features in train/val data.")

    # Combine them with target for scaling
    train_cols = common_features + [TARGET]
    val_cols   = common_features + [TARGET]

    # Scale using train min/max
    scaler = MinMaxScaler()

    # Fit scaler on train subset
    train_subset = train_df[train_cols].copy()
    train_scaled = scaler.fit_transform(train_subset)
    # Put back into train_df
    train_scaled_df = pd.DataFrame(train_scaled, columns=train_cols)
    train_df[train_cols] = train_scaled_df

    # Transform val subset
    val_subset = val_df[val_cols].copy()
    val_scaled = scaler.transform(val_subset)
    val_scaled_df = pd.DataFrame(val_scaled, columns=val_cols)
    val_df[val_cols] = val_scaled_df

    # Now create separate X, y
    train_X = train_df[common_features].values
    train_y = train_df[TARGET].values
    X_train, y_train = create_sequences(train_X, train_y, SEQ_LENGTH)

    val_X = val_df[common_features].values
    val_y = val_df[TARGET].values
    X_val, y_val = create_sequences(val_X, val_y, SEQ_LENGTH)

    print(f"Train sequences shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Val sequences shape:   X={X_val.shape}, y={y_val.shape}")

    if X_train.size == 0 or X_val.size == 0:
        raise ValueError("No train or val sequences created. Possibly dataset too small or SEQ_LENGTH too large.")

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

    ###################################
    # Phase 1: Random Search
    ###################################
    def random_build_model(hp):
        return build_model(hp, init_params=None)

    random_tuner = kt.RandomSearch(
        hypermodel=random_build_model,
        objective='val_loss',
        max_trials=10,  # or more
        directory='hvac_tuner_random_dir',
        project_name='random_search_hvac'
    )

    print("\n===== PHASE 1: RANDOM SEARCH =====")
    random_tuner.search(
        X_train, y_train,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # get best random-search hyperparams
    best_random_hp = random_tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest Hyperparameters from Random Search:")
    print(best_random_hp.values)

    best_random_model = random_tuner.get_best_models(num_models=1)[0]
    random_val_loss = best_random_model.evaluate(X_val, y_val, verbose=0)
    print(f"Random Search Best Model Val MSE: {random_val_loss:.4f}")

    ###################################
    # Phase 2: Bayesian Optimization
    ###################################
    def bayes_build_model(hp):
        return build_model(hp, init_params=best_random_hp.values)

    bayes_tuner = kt.BayesianOptimization(
        hypermodel=bayes_build_model,
        objective='val_loss',
        max_trials=10,
        num_initial_points=0,  # start from best_random_hp
        alpha=1e-4,
        beta=2.6,
        directory='hvac_tuner_bayes_dir',
        project_name='bayes_search_hvac'
    )

    print("\n===== PHASE 2: BAYESIAN OPTIMIZATION =====")
    bayes_tuner.search(
        X_train, y_train,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    best_bayes_hp = bayes_tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest Hyperparameters from Bayesian Optimization:")
    print(best_bayes_hp.values)

    best_bayes_model = bayes_tuner.get_best_models(num_models=1)[0]
    bayes_val_loss = best_bayes_model.evaluate(X_val, y_val, verbose=0)
    print(f"Bayes Best Model Val MSE: {bayes_val_loss:.4f}")

    # Compare
    print("\n===== SUMMARY =====")
    print(f"Random Search Best Val Loss:  {random_val_loss:.4f}")
    print(f"Bayes   Search Best Val Loss: {bayes_val_loss:.4f}")

    best_bayes_model.save("hvac_forecast_best_bayes.keras")
    print("✅ Final best model saved as 'hvac_forecast_best_bayes.keras'.")

if __name__ == "__main__":
    main()
