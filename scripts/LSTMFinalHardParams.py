import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Concatenate, SpatialDropout1D, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

###########################################
# 1) CONFIG
###########################################
DATA_FOLDER  = "/home/preben/Documents/Master/"
YR_FOLDER    = os.path.join(DATA_FOLDER, "yr")
VENT_FOLDER  = os.path.join(DATA_FOLDER, "vent")
ADAX_FOLDER  = os.path.join(DATA_FOLDER, "adax")
QLARM_FOLDER = os.path.join(DATA_FOLDER, "qlarm")

OUTPUT_CSV   = "merged_meetingroom_onlyEnergy_n2.csv"
MODEL_NAME   = "Thisisit_meetingroom_onlyEnergy_best_cpu_nr420.keras"

# Meeting room / target sensor names
MEETING_ROOM_ADAX_NAME    = "Meeting room 4th floor"
MEETING_ROOM_QLARM_T_COL  = "Preben - Meeting Room 434 Temperature"

# If you want a strict 5-min resample
RESAMPLE_5MIN = False

# Sequence config
SEQ_LENGTH = 24

# Data-split ratios
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2
TEST_RATIO  = 0.2

###########################################
# 2) FEATURES & TARGET
###########################################
BASE_FEATURES = [
    "energy_consumption",
    "supplyAirflow",
    "extractAirflow",
    "cloud_area_fraction",
    "dew_point_temperature",
    "air_temperature",
    "co2"
]
TARGET = MEETING_ROOM_QLARM_T_COL

# We will add three new columns for the “static” branch:
TIME_FEATURES = ["hour_of_day", "day_of_week", "is_weekend"]

###########################################
# 3) MERGE FUNCTIONS
###########################################
def load_json_files(folder_path):
    """Return a list of JSON objects from all .json files in folder_path."""
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

def merge_yr():
    """Merge all YR JSON files into a single DataFrame with columns:
         timestamp, [weather features...].
       The 'timestamp' is parsed from date_str + entry['time'].
    """
    dfs = []
    if not os.path.isdir(YR_FOLDER):
        print(f"⚠ YR folder not found: {YR_FOLDER}")
        return pd.DataFrame()

    for file in sorted(os.listdir(YR_FOLDER)):
        if file.endswith(".json"):
            # e.g. "yr2025-03-09.json" -> date_str="2025-03-09"
            date_str = file[2:12]
            file_path = os.path.join(YR_FOLDER, file)
            with open(file_path, "r") as f:
                data = json.load(f)
            # Build rows
            df = pd.DataFrame([
                {"timestamp": f"{date_str} {entry['time']}", **entry["data"]}
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

def merge_vent():
    """Merge all Vent JSON files into one DataFrame."""
    dfs = []
    if not os.path.isdir(VENT_FOLDER):
        print(f"⚠ Vent folder not found: {VENT_FOLDER}")
        return pd.DataFrame()

    vent_jsons = load_json_files(VENT_FOLDER)
    for data in vent_jsons:
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

def merge_adax_meeting_energy():
    """
    Keep only the meeting room from Adax.
    Return DataFrame with columns: [timestamp, energy_consumption].
    Summing 'energy' of all devices for that meeting room.
    """
    if not os.path.isdir(ADAX_FOLDER):
        print(f"⚠ Adax folder not found: {ADAX_FOLDER}")
        return pd.DataFrame()

    adax_files = load_json_files(ADAX_FOLDER)
    rows = []
    for data in adax_files:
        for entry in data:
            ts = entry.get("timestamp")
            if not ts:
                continue
            for room in entry.get("rooms", []):
                if room.get("name") == MEETING_ROOM_ADAX_NAME:
                    energy_sum = sum(d["energy"] for d in room.get("devices", []))
                    rows.append({
                        "timestamp": ts,
                        "energy_consumption": energy_sum
                    })

    if rows:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    return pd.DataFrame()

def merge_qlarm_meeting_temp():
    """
    Merge Qlarm CSVs, pivot them, and keep columns:
      [timestamp, MEETING_ROOM_QLARM_T_COL]
    """
    if not os.path.isdir(QLARM_FOLDER):
        print(f"⚠ Qlarm folder not found: {QLARM_FOLDER}")
        return pd.DataFrame()

    dfs = []
    for file in sorted(os.listdir(QLARM_FOLDER)):
        if file.endswith(".csv"):
            file_path = os.path.join(QLARM_FOLDER, file)
            df = pd.read_csv(file_path)
            if "Time" in df.columns:
                df.rename(columns={"Time": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # pivot the data so that sensor names become columns
            df = df.pivot_table(index="timestamp", 
                                columns="SensorName", 
                                values="Value", 
                                aggfunc="mean").reset_index()
            dfs.append(df)

    if dfs:
        out = pd.concat(dfs, ignore_index=True)
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)

        keep_cols = ["timestamp"]
        if MEETING_ROOM_QLARM_T_COL in out.columns:
            keep_cols.append(MEETING_ROOM_QLARM_T_COL)
        else:
            print(f"⚠ Target '{MEETING_ROOM_QLARM_T_COL}' not found in Qlarm pivot.")
        out = out[keep_cols]
        return out
    return pd.DataFrame()

def merge_all_and_save():
    """
    Merge YR, Vent, Adax (meeting room), Qlarm (meeting room temp).
    Forward-fill missing values. Optionally resample to 5 minutes.
    Save to OUTPUT_CSV, then return the final merged DataFrame.
    """
    yr_df   = merge_yr()
    vent_df = merge_vent()
    adax_df = merge_adax_meeting_energy()
    qlarm_df= merge_qlarm_meeting_temp()

    print(f"YR shape: {yr_df.shape}")
    print(f"Vent shape: {vent_df.shape}")
    print(f"Adax shape: {adax_df.shape}")
    print(f"Qlarm shape: {qlarm_df.shape}")

    # Merge on timestamp using outer joins
    merged_df = (
        adax_df.merge(vent_df, on="timestamp", how="outer")
               .merge(yr_df,   on="timestamp", how="outer")
               .merge(qlarm_df,on="timestamp", how="outer")
    )
    merged_df.sort_values("timestamp", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.ffill(inplace=True)  # forward fill

    if RESAMPLE_5MIN:
        merged_df.set_index("timestamp", inplace=True)
        merged_df = merged_df.resample("5T").ffill()
        merged_df.reset_index(inplace=True)
        merged_df.sort_values("timestamp", inplace=True)

    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved merged data to '{OUTPUT_CSV}' with shape={merged_df.shape}")
    return merged_df

###########################################
# 4) CREATE SEQUENCES
###########################################
def create_sequences_with_time(dataX, dataY, df_time_cols, seq_length):
    """
    dataX: 2D numpy of shape [N, len(BASE_FEATURES)]
    dataY: 1D numpy of shape [N]
    df_time_cols: 2D numpy for time-based columns [N, len(TIME_FEATURES)]
    seq_length: how many steps in each input sequence

    Returns:
      X_seq: shape [M, seq_length, len(BASE_FEATURES)]
      X_static: shape [M, len(TIME_FEATURES)]   # single static vector per sequence
      y_seq: shape [M]
    """
    X_seq, X_static, y_seq = [], [], []
    for i in range(len(dataX) - seq_length):
        seq_x = dataX[i : i + seq_length]
        seq_y = dataY[i + seq_length]  # predict the "next" time step
        t_feats = df_time_cols[i + seq_length]  # static features at the future step

        # Only add sequence if no NaNs exist
        if not np.isnan(seq_x).any() and not np.isnan(seq_y):
            X_seq.append(seq_x)
            X_static.append(t_feats)
            y_seq.append(seq_y)

    X_seq = np.array(X_seq, dtype=np.float32)
    X_static = np.array(X_static, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)
    return X_seq, X_static, y_seq

###########################################
# 5) BUILD MODEL (Fixed Hyperparameters)
###########################################
def build_sequence_model():
    """
    Build an LSTM model with a parallel static input branch using fixed hyperparameters.
    Hyperparameters are set as:
      - rnn_type: LSTM
      - num_rnn_layers: 1
      - units_1: 416
      - dropout_1: 0.2
      - rec_dropout_1: 0.0
      - l2_reg: 1e-05
      - use_spatial_dropout: True
      - use_layernorm: False
      - static_dense_units: 32
      - final_dense_units: 16
      - learning_rate: 0.0001
      - (For 2-layer configurations, units_2: 64, dropout_2: 0.0, rec_dropout_2: 0.0)
    """
    # Fixed hyperparameters
    rnn_type = 'LSTM'
    num_rnn_layers = 1
    units_1 = 416
    dropout_1 = 0.2
    rec_dropout_1 = 0.0
    l2_reg = 1e-05
    use_spatial_dropout = True
    use_layernorm = False
    static_dense_units = 32
    final_dense_units = 16
    learning_rate = 0.0001
    # The following parameters are for a second layer (not used since num_rnn_layers == 1)
    units_2 = 64
    dropout_2 = 0.0
    rec_dropout_2 = 0.0

    # Two input layers:
    # 1) time_series_input: shape=(SEQ_LENGTH, len(BASE_FEATURES))
    # 2) static_input: shape=(len(TIME_FEATURES),)
    time_series_input = Input(shape=(SEQ_LENGTH, len(BASE_FEATURES)), name="time_series_input")
    static_input = Input(shape=(len(TIME_FEATURES),), name="static_input")

    # Optionally apply SpatialDropout1D before RNN
    x = time_series_input
    if use_spatial_dropout:
        x = SpatialDropout1D(0.1)(x)

    # Build the RNN stack (only one layer in this configuration)
    if rnn_type == 'LSTM':
        x = LSTM(units_1,
                 dropout=dropout_1,
                 recurrent_dropout=rec_dropout_1,
                 kernel_regularizer=regularizers.l2(l2_reg),
                 return_sequences=False)(x)
    else:
        x = GRU(units_1,
                dropout=dropout_1,
                recurrent_dropout=rec_dropout_1,
                kernel_regularizer=regularizers.l2(l2_reg),
                return_sequences=False)(x)

    # Optionally apply layer normalization after RNN
    if use_layernorm:
        x = LayerNormalization()(x)

    # Static branch for time-based features
    s = Dense(static_dense_units, activation='relu',
              kernel_regularizer=regularizers.l2(l2_reg))(static_input)
    s = Dense(static_dense_units // 2, activation='relu',
              kernel_regularizer=regularizers.l2(l2_reg))(s)

    # Merge RNN and static branches
    merged = Concatenate()([x, s])

    # Final dense layers
    out = Dense(final_dense_units, activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg))(merged)
    out = Dense(1)(out)

    model = Model(inputs=[time_series_input, static_input], outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='mae',
        metrics=['mae', 'mse']
    )
    return model

###########################################
# 6) EVALUATION FUNCTIONS
###########################################
def evaluate_model(model, X_test_seq, X_test_static, y_test):
    """
    Evaluate model on test set (scaled evaluation).
    """
    preds = model.predict([X_test_seq, X_test_static]).flatten()
    mae  = mean_absolute_error(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    rmse = sqrt(mse)
    r2   = r2_score(y_test, preds)

    print("\n===== TEST SET PERFORMANCE (Scaled) =====")
    print(f"MAE  = {mae:.4f}")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R^2  = {r2:.4f}")

    n_points = min(500, len(y_test))
    plt.figure()
    plt.plot(y_test[-n_points:], label="Actual (scaled)")
    plt.plot(preds[-n_points:], label="Predicted (scaled)")
    plt.title("Meeting Room 434: Last 500 Test Steps (Scaled Values)")
    plt.xlabel("Time Step")
    plt.ylabel("Scaled Temp")
    plt.legend()
    plt.show()

    plt.figure()
    plt.scatter(y_test, preds, alpha=0.3)
    plt.title("Predicted vs. Actual (Scaled)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

    residuals = y_test - preds
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution (Scaled)")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Count")
    plt.show()

def evaluate_model_unscaled(model, X_test_seq, X_test_static, y_test, scaler_y):
    """
    Evaluate model on test set by inverse transforming the predictions and actual values
    using scaler_y, then plotting the unscaled predicted vs. actual temperatures.
    """
    preds = model.predict([X_test_seq, X_test_static]).flatten()
    preds_unscaled = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae  = mean_absolute_error(y_test_unscaled, preds_unscaled)
    mse  = mean_squared_error(y_test_unscaled, preds_unscaled)
    rmse = sqrt(mse)
    r2   = r2_score(y_test_unscaled, preds_unscaled)

    print("\n===== TEST SET PERFORMANCE (Unscaled) =====")
    print(f"MAE  = {mae:.4f}")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R^2  = {r2:.4f}")

    n_points = min(500, len(y_test_unscaled))
    plt.figure()
    plt.plot(y_test_unscaled[-n_points:], label="Actual Temperature")
    plt.plot(preds_unscaled[-n_points:], label="Predicted Temperature")
    plt.title("Temperature Prediction: Last 500 Test Steps (Unscaled)")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()

    plt.figure()
    plt.scatter(y_test_unscaled, preds_unscaled, alpha=0.3)
    plt.title("Predicted vs. Actual Temperature (Unscaled)")
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.show()

    residuals = y_test_unscaled - preds_unscaled
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution (Unscaled Temperature)")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Count")
    plt.show()

###########################################
# 7) MAIN
###########################################
def main():
    # 1) Merge all data
    merged_df = merge_all_and_save()

    # 2) Check if the target column is present
    if TARGET not in merged_df.columns:
        raise ValueError(f"Target '{TARGET}' not found in merged DataFrame columns.")

    # 3) Create time-based features
    merged_df['hour_of_day'] = merged_df['timestamp'].dt.hour
    merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
    merged_df['is_weekend']  = merged_df['day_of_week'].isin([5,6]).astype(int)

    # 4) Filter columns to use
    all_columns = BASE_FEATURES + [TARGET] + TIME_FEATURES
    missing_cols = [c for c in all_columns if c not in merged_df.columns]
    if missing_cols:
        print(f"⚠ Missing columns in merged data: {missing_cols}")

    used_df = merged_df[all_columns].copy()

    # 5) Scaling: Use separate scalers for features and target.
    features_columns = BASE_FEATURES + TIME_FEATURES
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    used_df[features_columns] = scaler_features.fit_transform(used_df[features_columns])
    used_df[TARGET] = scaler_target.fit_transform(used_df[[TARGET]])

    # 6) Create sequences
    dataX = used_df[BASE_FEATURES].values
    dataY = used_df[TARGET].values
    timeData = used_df[TIME_FEATURES].values

    X_seq, X_static, y_seq = create_sequences_with_time(dataX, dataY, timeData, SEQ_LENGTH)
    print(f"\nSequence data shapes -> X_seq={X_seq.shape}, X_static={X_static.shape}, y_seq={y_seq.shape}")

    # 7) Train/Val/Test split (time-based)
    n = len(X_seq)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    X_train_seq = X_seq[:train_end]
    X_val_seq = X_seq[train_end:val_end]
    X_test_seq = X_seq[val_end:]
    X_train_static = X_static[:train_end]
    X_val_static = X_static[train_end:val_end]
    X_test_static = X_static[val_end:]
    y_train = y_seq[:train_end]
    y_val = y_seq[train_end:val_end]
    y_test = y_seq[val_end:]

    print("Train:", X_train_seq.shape, X_train_static.shape, y_train.shape)
    print("Val:  ", X_val_seq.shape, X_val_static.shape, y_val.shape)
    print("Test: ", X_test_seq.shape, X_test_static.shape, y_test.shape)

    # 8) Build and train the model with fixed hyperparameters
    model = build_sequence_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

    print("\n===== Starting Training =====")
    history = model.fit(
        [X_train_seq, X_train_static], y_train,
        epochs=50,
        validation_data=([X_val_seq, X_val_static], y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # 9) Evaluate on test set and save the model
    evaluate_model_unscaled(model, X_test_seq, X_test_static, y_test, scaler_target)
    model.save(MODEL_NAME)
    print(f"✅ Model saved as '{MODEL_NAME}'.")

if __name__ == "__main__":
    main()
