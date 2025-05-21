import os
import sys
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Ensure proper UTF-8 encoding to prevent UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

# Import GRU instead of LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ✅ Ensure TensorFlow Uses GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ TensorFlow is using GPU (FAST MODE)")
    except RuntimeError as e:
        print(f"⚠️ GPU Configuration Error: {e}")
else:
    print("⚠️ No GPU detected. Running on CPU.")

# ✅ Define Paths
DATA_FOLDER = "/home/preben/Documents/Master/"  # Update with your actual path
folders = ["yr", "vent", "adax", "qlarm"]  # Folders containing data

# ✅ Function to Load JSON Files (for Vent and Adax)
def load_json_files(folder_path):
    data_list = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data_list.append(json.load(f))
    return data_list

print("🔄 Loading data from multiple files...")

# ✅ Load and Process Data
def load_and_process_data():
    # Load data
    yr_data, vent_data, adax_data, qlarm_data = [load_json_files(os.path.join(DATA_FOLDER, f)) for f in folders]
    
    # Process YR Weather Forecast Data
    yr_dfs = []
    for data in yr_data:
        df = pd.DataFrame([{"timestamp": pd.to_datetime(entry.get("timestamp") or entry.get("time") or entry.get("datetime"), errors='coerce'), **entry.get("data", {})} for entry in data])
        df.dropna(subset=["timestamp"], inplace=True)
        yr_dfs.append(df)
    yr_df = pd.concat(yr_dfs, ignore_index=True) if yr_dfs else pd.DataFrame()
    
    # Process Ventilation Data
    vent_dfs = [pd.DataFrame(data) for data in vent_data]
    for df in vent_dfs:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    vent_df = pd.concat(vent_dfs, ignore_index=True) if vent_dfs else pd.DataFrame()
    
    # Process Adax Heating Panel Data
    adax_dfs = []
    for data in adax_data:
        temp_list = []
        for entry in data:
            timestamp = pd.to_datetime(entry.get("timestamp"), errors='coerce')
            for room in entry.get("rooms", []):
                temp_list.append({"timestamp": timestamp, "room_id": room.get("id"), "room_name": room.get("name"), "target_temp": room.get("targetTemperature"), "current_temp": room.get("currentTemperature"), "energy_consumption": sum(dev.get("energy", 0) for dev in room.get("devices", []))})
        df = pd.DataFrame(temp_list)
        adax_dfs.append(df)
    adax_df = pd.concat(adax_dfs, ignore_index=True) if adax_dfs else pd.DataFrame()
    
    # Process Qlarm Data
    qlarm_dfs = []
    for file in os.listdir(os.path.join(DATA_FOLDER, "qlarm")):
        if file.endswith(".csv"):
            file_path = os.path.join(DATA_FOLDER, "qlarm", file)
            df = pd.read_csv(file_path)
            df.rename(columns={"Time": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            df = df.pivot_table(index="timestamp", columns="SensorName", values="Value", aggfunc='mean').reset_index()
            qlarm_dfs.append(df)
    qlarm_df = pd.concat(qlarm_dfs, ignore_index=True) if qlarm_dfs else pd.DataFrame()
    
    # Ensure all timestamps are datetime type before merging
    for df in [adax_df, vent_df, yr_df, qlarm_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    
    # Merge All Datasets
    merged_df = pd.concat([adax_df, vent_df, yr_df, qlarm_df], axis=0).sort_values(by="timestamp").reset_index(drop=True)
    merged_df.ffill(inplace=True)
    
    return merged_df

# ✅ Load Data
merged_df = load_and_process_data()
print(f"✅ Merged dataset saved with {len(merged_df)} entries.")

# ✅ Define Features and Target
features = merged_df.columns.difference(["timestamp"]).tolist()
target = "current_temp"

# ✅ Normalize Data
scaler = MinMaxScaler()
merged_df[features] = scaler.fit_transform(merged_df[features])

# ✅ Train-Test Split
SEQ_LENGTH = 24
X, y = [], []
for i in range(len(merged_df) - SEQ_LENGTH):
    X.append(merged_df[features].iloc[i:i+SEQ_LENGTH].values)
    y.append(merged_df[target].iloc[i+SEQ_LENGTH])
X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ✅ Define and Train Model
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(SEQ_LENGTH, len(features)), dropout=0.2, recurrent_dropout=0.2),
    GRU(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# ✅ Training with Progress Bar
with tqdm(total=100, desc="Training Progress", unit="epoch") as pbar:
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)], verbose=1)
    pbar.update(100)

# ✅ Save Model
model.save("gru_hvac_forecast.keras")
print("✅ Trained GRU model saved as 'gru_hvac_forecast.keras'.")