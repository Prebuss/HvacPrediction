import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ✅ Additional imports for regularization and callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ✅ Ensure TensorFlow Uses GPU
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

# ✅ Define Paths
DATA_FOLDER = "/home/preben/Documents/Master/"  # Update with your actual path
folders = ["selectiveYr", "selectiveVent", "selectiveAdax", "selectiveQlarm"]  # Folders containing data

# ✅ Function to Load JSON Files (for Vent and Adax)
def load_json_files(folder_path):
    data_list = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                data_list.append(json.load(f))
    return data_list

print("🔄 Loading data from multiple files...")

# ✅ Process YR Weather Forecast Data
# Extract the date from the filename (assuming format "yrYYYY-MM-DD.json")
yr_dfs = []
yr_folder = os.path.join(DATA_FOLDER, "yr")
for file in sorted(os.listdir(yr_folder)):
    if file.endswith(".json"):
        file_path = os.path.join(yr_folder, file)
        # Extract date from filename; adjust slicing if needed.
        date_str = file[2:12]  # For "yr2025-03-09.json", this extracts "2025-03-09"
        with open(file_path, "r") as f:
            data = json.load(f)
        # Create DataFrame by combining extracted date with the "time" field
        df = pd.DataFrame([
            {"timestamp": f"{date_str} {entry['time']}", **entry["data"]}
            for entry in data
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        yr_dfs.append(df)
yr_df = pd.concat(yr_dfs, ignore_index=True) if yr_dfs else pd.DataFrame()

# ✅ Process Ventilation Data
vent_folder = os.path.join(DATA_FOLDER, "vent")
vent_data = load_json_files(vent_folder)
vent_dfs = []
for data in vent_data:
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    vent_dfs.append(df)
vent_df = pd.concat(vent_dfs, ignore_index=True) if vent_dfs else pd.DataFrame()

# ✅ Process Adax Heating Panel Data
adax_folder = os.path.join(DATA_FOLDER, "adax")
adax_data = load_json_files(adax_folder)
adax_dfs = []
for data in adax_data:
    temp_list = []
    for entry in data:
        timestamp = entry["timestamp"]
        for room in entry["rooms"]:
            temp_list.append({
                "timestamp": timestamp,
                "room_id": room["id"],
                "room_name": room["name"],
                "target_temp": room["targetTemperature"],
                "current_temp": room["currentTemperature"],
                "energy_consumption": sum(dev["energy"] for dev in room["devices"])
            })
    df = pd.DataFrame(temp_list)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    adax_dfs.append(df)
adax_df = pd.concat(adax_dfs, ignore_index=True) if adax_dfs else pd.DataFrame()

# ✅ Process Qlarm Data (CSV files)
qlarm_folder = os.path.join(DATA_FOLDER, "qlarm")
qlarm_dfs = []
for file in os.listdir(qlarm_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(qlarm_folder, file)
        df = pd.read_csv(file_path)
        df.rename(columns={"Time": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Use pivot_table with an aggregation function to handle duplicate entries
        df = df.pivot_table(index="timestamp", columns="SensorName", values="Value", aggfunc="mean").reset_index()
        qlarm_dfs.append(df)
qlarm_df = pd.concat(qlarm_dfs, ignore_index=True) if qlarm_dfs else pd.DataFrame()

# ✅ Merge All Datasets
merged_df = (
    adax_df.merge(vent_df, on="timestamp", how="outer")
           .merge(yr_df, on="timestamp", how="outer")
           .merge(qlarm_df, on="timestamp", how="outer")
)

# ✅ Sort and Fill Missing Values
merged_df.sort_values(by="timestamp", inplace=True)
merged_df.ffill(inplace=True)  # Fill missing values with last known

# ✅ Save Merged Dataset
merged_df.to_csv("merged_hvac_dataset.csv", index=False)
print(f"✅ Merged dataset saved with {len(merged_df)} entries.")

# ✅ Select Features for Training
features = [
    "target_temp", "current_temp", "energy_consumption",
    "co2", "supplyAirflow", "extractAirflow", "supplyAirDuctPressure",
    "extractAirDuctPressure", "supplyAirFanSpeedLevel", "extractAirFanSpeedLevel",
    "supplyAirTemperature", "extractAirTemperature", "outdoorTemperature",
    "reheatLevel", "coolingLevel", "heatExchangerRegulator", "RhxEfficiency",
    "air_temperature", "relative_humidity", "wind_speed"
]
target = "current_temp"

# ✅ Normalize Data
scaler = MinMaxScaler()
merged_df[features] = scaler.fit_transform(merged_df[features])

# ✅ Define Sequence Length
SEQ_LENGTH = 24

# ✅ Create Sequences
def create_sequences(data, target_column, seq_length):
    X, y = [], []
    target_idx = features.index(target_column)
    for i in range(len(data) - seq_length):
        seq_X = data[i:i+seq_length]
        seq_y = data[i+seq_length, target_idx]
        # Only add sequence if no NaNs in features or target
        if np.isnan(seq_X).sum() == 0 and not np.isnan(seq_y):
            X.append(seq_X)
            y.append(seq_y)
    return np.array(X), np.array(y)

X, y = create_sequences(merged_df[features].values, target, SEQ_LENGTH)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ========== MODEL IMPROVEMENTS BEGIN HERE ========== #

# Build LSTM Model with updated hyperparameters from autotuner
model = Sequential([
    LSTM(
        128,  # units_1
        return_sequences=True,
        input_shape=(SEQ_LENGTH, len(features)),
        dropout=0.1,            # dropout_1
        recurrent_dropout=0.0   # rec_dropout_1
    ),
    LSTM(
        112,  # units_2
        return_sequences=False,
        dropout=0.0,            # dropout_2
        recurrent_dropout=0.2   # rec_dropout_2
    ),
    Dense(
        16,   # dense_units
        activation="relu",
        kernel_regularizer=regularizers.l2(0.0006764133576864393)  # l2_reg
    ),
    Dense(1)  # Predict future temperature
])

# Use the autotuner's learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002513168195838526)
model.compile(optimizer=optimizer, loss="mse")


# ✅ Callbacks for Early Stopping and Learning Rate Reduction
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,            # Stop after 5 epochs with no improvement
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,            # Reduce the LR by a factor of 5
    patience=3,            # Wait 3 epochs before reducing
    min_lr=1e-6            # Minimum possible LR
)

# ✅ Train Model (with callbacks)
history = model.fit(
    X_train, y_train,
    epochs=100,            # Increased epochs to allow early stopping
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ✅ Save Model
model.save("lstm_hvac_forecast.keras")
print("✅ Trained model saved as 'lstm_hvac_forecast.keras'.")

# ✅ Evaluate Model
test_loss = model.evaluate(X_test, y_test)
print(f"📉 Test Loss (MSE): {test_loss}")

# ✅ Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("LSTM HVAC Training Loss Over Epochs")
plt.show()
