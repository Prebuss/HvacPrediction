import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

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

# ✅ Load YR Weather Forecast Data
with open("/home/preben/Documents/Master/yr/yr2025-02-20.json", "r") as file:
    yr_data = json.load(file)

yr_df = pd.DataFrame([
    {"timestamp": f"2025-02-20 {entry['time']}", **entry["data"]}
    for entry in yr_data
])
yr_df["timestamp"] = pd.to_datetime(yr_df["timestamp"])

# ✅ Load Ventilation Data
with open("/home/preben/Documents/Master/vent/newVentilation2025-02-20.json", "r") as file:
    vent_data = json.load(file)

vent_df = pd.DataFrame(vent_data)
vent_df["timestamp"] = pd.to_datetime(vent_df["timestamp"])

# ✅ Load Adax Heating Panel Data
with open("/home/preben/Documents/Master/adax/2025-02-20.json", "r") as file:
    adax_data = json.load(file)

adax_list = []
for entry in adax_data:
    timestamp = entry["timestamp"]
    for room in entry["rooms"]:
        adax_list.append({
            "timestamp": timestamp,
            "room_id": room["id"],
            "room_name": room["name"],
            "target_temp": room["targetTemperature"],
            "current_temp": room["currentTemperature"],
            "energy_consumption": sum(dev["energy"] for dev in room["devices"])
        })

adax_df = pd.DataFrame(adax_list)
adax_df["timestamp"] = pd.to_datetime(adax_df["timestamp"])

# ✅ Load Qlarm Data
qlarm_df = pd.read_csv("/home/preben/Documents/Master/qlarm/qlarmData 2025.02.20.csv")

# Ensure Qlarm has correct timestamp column
qlarm_df.rename(columns={"Time": "timestamp"}, inplace=True)
qlarm_df["timestamp"] = pd.to_datetime(qlarm_df["timestamp"])

# Pivot the Qlarm dataset so each sensor type becomes a separate column
qlarm_df = qlarm_df.pivot(index="timestamp", columns="SensorName", values="Value").reset_index()

# ✅ Merge all datasets on timestamp
merged_df = (
    adax_df.merge(vent_df, on="timestamp", how="outer")
           .merge(yr_df, on="timestamp", how="outer")
           .merge(qlarm_df, on="timestamp", how="outer")
)

# ✅ Sort and fill missing values
merged_df.sort_values(by="timestamp", inplace=True)

# Check for NaN values before filling
if merged_df.isnull().sum().sum() > 0:
    print("⚠️ Dataset contains NaN values! Filling missing values...")
    merged_df.ffill(inplace=True)

# ✅ Save merged dataset
merged_df.to_csv("merged_hvac_dataset.csv", index=False)
print("✅ Merged dataset saved as 'merged_hvac_dataset.csv'.")

# ✅ Select features for LSTM Model
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

# ✅ Create time series sequences
SEQ_LENGTH = 24  # Use past 24 timesteps for prediction

def create_sequences(data, target_column, seq_length):
    """Generate sequences for LSTM model training with NaN checks."""
    X, y = [], []
    target_idx = features.index(target_column)
    
    for i in range(len(data) - seq_length):
        seq_X = data[i:i+seq_length]
        seq_y = data[i+seq_length, target_idx]
        
        if np.isnan(seq_X).sum() == 0 and not np.isnan(seq_y):
            X.append(seq_X)
            y.append(seq_y)
    
    return np.array(X), np.array(y)

X, y = create_sequences(merged_df[features].values, target, SEQ_LENGTH)

# ✅ Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ✅ Build LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, len(features))),
    LSTM(64, return_sequences=False),
    Dense(32, activation="relu"),
    Dense(1)  # Predict future room temperature
])

# ✅ Compile Model
model.compile(optimizer="adam", loss="mse")

# ✅ Train Model & Track History
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# ✅ Save Trained Model
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

# ✅ Make Predictions
last_sequence = X_test[-1].reshape(1, SEQ_LENGTH, len(features))
predicted_temp = model.predict(last_sequence)
predicted_temp = scaler.inverse_transform([[predicted_temp[0][0]]])[0][0]

print(f"🔮 Predicted Future Room Temperature: {predicted_temp:.2f}°C")

# ✅ Compare Predictions vs Actual
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.xlabel("Time Steps")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.title("LSTM HVAC Predictions vs Actual")
plt.show()
