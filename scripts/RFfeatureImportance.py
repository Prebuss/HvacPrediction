import os
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # optional for progress bars

# ========== 1) SETUP & DATA LOADING ========== #

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

# --- Load Ventilation Data ---
vent_folder = os.path.join(DATA_FOLDER, "vent")
vent_dfs = []
if os.path.isdir(vent_folder):
    vent_data = load_json_files(vent_folder)
    for data in vent_data:
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        vent_dfs.append(df)
vent_df = pd.concat(vent_dfs, ignore_index=True) if vent_dfs else pd.DataFrame()

# --- Load Adax Data ---
adax_folder = os.path.join(DATA_FOLDER, "adax")
adax_dfs = []
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

# --- Load YR (Weather) ---
yr_folder = os.path.join(DATA_FOLDER, "yr")
yr_dfs = []
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

# --- Load Qlarm CSV Data ---
qlarm_folder = os.path.join(DATA_FOLDER, "qlarm")
qlarm_dfs = []
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

# ========== 2) MERGE ALL DFs ========== #
merged_df = (
    adax_df.merge(vent_df, on="timestamp", how="outer")
           .merge(yr_df, on="timestamp", how="outer")
           .merge(qlarm_df, on="timestamp", how="outer")
)

merged_df.sort_values("timestamp", inplace=True)
merged_df.ffill(inplace=True)  # Fill forward any missing

merged_df.to_csv("merged_hvac_dataset.csv", index=False)
print(f"✅ Merged dataset saved with {len(merged_df)} entries.")
print("Columns in merged_df:", merged_df.columns.tolist())

# ========== 3) SELECT FEATURES & TARGET ========== #
target = "target_temp"

# Potential features you suspect might influence "target_temp"
all_candidate_features = [
    "current_temp",
    "energy_consumption",
    "co2",
    "supplyAirflow", "extractAirflow",
    "supplyAirTemperature", "extractAirTemperature",
    "outdoorTemperature",
    # etc. add more if you want
]

# Filter out any that don't exist in the merged DataFrame
all_candidate_features = [f for f in all_candidate_features if f in merged_df.columns]

# Drop rows where target is missing
df_model = merged_df.dropna(subset=[target])

# If you want to also drop rows where features are missing, do:
# df_model = df_model.dropna(subset=all_candidate_features)

# Final X,y
X = df_model[all_candidate_features]
y = df_model[target]

# ========== 4) SPLIT & TRAIN RANDOM FOREST ========== #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

rf = RandomForestRegressor(
    n_estimators=100,       # how many trees
    max_depth=10,           # limit depth to avoid huge trees
    random_state=42
)
rf.fit(X_train, y_train)

score = rf.score(X_test, y_test)
print(f"Random Forest R^2 on test set: {score:.4f}")

# ========== 5) GET FEATURE IMPORTANCE ========== #
importances = rf.feature_importances_  # array of shape (#features,)
feature_importance_list = sorted(
    zip(all_candidate_features, importances),
    key=lambda x: x[1],
    reverse=True
)

print("\n=== Feature Importances (Random Forest) ===")
for feat, imp in feature_importance_list:
    print(f"{feat:30s}  importance={imp:.4f}")
