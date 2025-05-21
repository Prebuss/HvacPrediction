import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# ✅ Define Paths
DATA_FOLDER = "/home/preben/Documents/Master/"  # Update with your actual path
folders = ["selectiveYr", "selectiveVent", "selectiveAdax", "selectiveQlarm"]

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
yr_dfs = []
yr_folder = os.path.join(DATA_FOLDER, "yr")
for file in sorted(os.listdir(yr_folder)):
    if file.endswith(".json"):
        file_path = os.path.join(yr_folder, file)
        # Extract date from filename; adjust slicing if needed
        date_str = file[2:12]  # For "yr2025-03-09.json", extracts "2025-03-09"
        with open(file_path, "r") as f:
            data = json.load(f)
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

# ✅ Define Sequence Length (number of time steps in the sliding window)
SEQ_LENGTH = 24

# ✅ Create Flattened Feature Vectors from Sliding Windows
def create_flattened_features_and_target(data, target_column, seq_length):
    X, y = [], []
    target_idx = features.index(target_column)
    for i in range(len(data) - seq_length):
        # Extract a window of SEQ_LENGTH time steps and flatten it
        window = data[i:i+seq_length]
        flat_window = window.flatten()  # Resulting shape: (seq_length * num_features,)
        target_value = data[i+seq_length, target_idx]  # The target is the value at the next time step
        # Append only if there are no missing values
        if np.isnan(flat_window).sum() == 0 and not np.isnan(target_value):
            X.append(flat_window)
            y.append(target_value)
    return np.array(X), np.array(y)

# Prepare the features and target arrays
X, y = create_flattened_features_and_target(merged_df[features].values, target, SEQ_LENGTH)

# ✅ Train-Test Split (no shuffle for time series data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ✅ Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ✅ Evaluate the Model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"📉 Test MSE: {mse}")

# ✅ Save the Random Forest Model
joblib.dump(rf_model, "rf_hvac_forecast_model.joblib")
print("✅ Random Forest model saved as 'rf_hvac_forecast_model.joblib'.")

# ✅ Optional: Plot Actual vs. Predicted Values
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Predicted", alpha=0.7)
plt.xlabel("Time Index")
plt.ylabel("Normalized Current Temperature")
plt.title("Random Forest HVAC Forecast: Actual vs. Predicted")
plt.legend()
plt.show()
