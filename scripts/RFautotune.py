import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import joblib

# ============================
# Data Loading and Processing
# ============================

DATA_FOLDER = "/home/preben/Documents/Master/"  # Update with your actual path

def load_json_files(folder_path):
    data_list = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                data_list.append(json.load(f))
    return data_list

print("🔄 Loading data from multiple files...")

# --- Process YR Weather Forecast Data ---
yr_dfs = []
yr_folder = os.path.join(DATA_FOLDER, "yr")
for file in sorted(os.listdir(yr_folder)):
    if file.endswith(".json"):
        file_path = os.path.join(yr_folder, file)
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

# --- Process Ventilation Data ---
vent_folder = os.path.join(DATA_FOLDER, "vent")
vent_data = load_json_files(vent_folder)
vent_dfs = []
for data in vent_data:
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    vent_dfs.append(df)
vent_df = pd.concat(vent_dfs, ignore_index=True) if vent_dfs else pd.DataFrame()

# --- Process Adax Heating Panel Data ---
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

# --- Process Qlarm Data (CSV files) ---
qlarm_folder = os.path.join(DATA_FOLDER, "qlarm")
qlarm_dfs = []
for file in os.listdir(qlarm_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(qlarm_folder, file)
        df = pd.read_csv(file_path)
        df.rename(columns={"Time": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Pivot to handle duplicate entries
        df = df.pivot_table(index="timestamp", columns="SensorName", values="Value", aggfunc="mean").reset_index()
        qlarm_dfs.append(df)
qlarm_df = pd.concat(qlarm_dfs, ignore_index=True) if qlarm_dfs else pd.DataFrame()

# --- Merge All Datasets ---
merged_df = (
    adax_df.merge(vent_df, on="timestamp", how="outer")
           .merge(yr_df, on="timestamp", how="outer")
           .merge(qlarm_df, on="timestamp", how="outer")
)
merged_df.sort_values(by="timestamp", inplace=True)
merged_df.ffill(inplace=True)
# Also fill any remaining missing values using backward fill:
merged_df.fillna(method='bfill', inplace=True)
merged_df.to_csv("merged_hvac_dataset.csv", index=False)
print(f"✅ Merged dataset saved with {len(merged_df)} entries.")

# ========================
# Feature Selection & Prep
# ========================

features = [
    "target_temp", "current_temp", "energy_consumption",
    "co2", "supplyAirflow", "extractAirflow", "supplyAirDuctPressure",
    "extractAirDuctPressure", "supplyAirFanSpeedLevel", "extractAirFanSpeedLevel",
    "supplyAirTemperature", "extractAirTemperature", "outdoorTemperature",
    "reheatLevel", "coolingLevel", "heatExchangerRegulator", "RhxEfficiency",
    "air_temperature", "relative_humidity", "wind_speed"
]
target = "current_temp"

# Normalize features
scaler = MinMaxScaler()
merged_df[features] = scaler.fit_transform(merged_df[features])

# Prepare target variable by shifting (predict next timestamp's value)
merged_df['target_next'] = merged_df[target].shift(-1)
merged_df.dropna(subset=['target_next'], inplace=True)

X = merged_df[features]
y = merged_df['target_next']

# Train-Test Split (preserving time order)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ============================================
# Build Pipeline & Expanded Hyperparameter Grid
# ============================================

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('pca', PCA(n_components=0.95)),  # Retain 95% of variance
    ('rf', RandomForestRegressor(random_state=42))
])

# Use only valid criterion values: 'squared_error' and 'absolute_error'
param_grid_pipeline = {
    'rf__n_estimators': [50, 100, 200, 500, 750],
    'rf__max_depth': [None, 5, 10, 20, 30, 50],
    'rf__min_samples_split': [2, 5, 10, 15],
    'rf__min_samples_leaf': [1, 2, 4, 6],
    'rf__max_features': ['auto', 'sqrt', 'log2'],
    'rf__criterion': ['squared_error', 'absolute_error'],
    'rf__bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid_pipeline,
    n_iter=50,  # Increased number of iterations
    scoring='neg_mean_squared_error',
    cv=5,       # 5-fold CV for inner loop
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# ========================
# Nested Cross-Validation
# ========================

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_scores = []

print("🔍 Starting nested cross-validation on training data...")
for train_ix, val_ix in outer_cv.split(X_train):
    X_tr, X_val = X_train.iloc[train_ix], X_train.iloc[val_ix]
    y_tr, y_val = y_train.iloc[train_ix], y_train.iloc[val_ix]
    
    # Hyperparameter tuning on inner training set
    random_search.fit(X_tr, y_tr)
    best_model = random_search.best_estimator_
    preds = best_model.predict(X_val)
    score = mean_squared_error(y_val, preds)
    outer_scores.append(score)

print("Nested CV MSE scores:", outer_scores)
print("Mean Nested CV MSE:", np.mean(outer_scores))

# ========================================
# Final Model Training on Entire Training Set
# ========================================

print("🔍 Retraining final model on the full training set...")
random_search.fit(X_train, y_train)
best_model_final = random_search.best_estimator_
y_pred_test = best_model_final.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
print(f"Test MSE on holdout set: {test_mse}")

# Save the best model
joblib.dump(best_model_final, "rf_hvac_forecast_best_complex.pkl")
print("✅ Best model saved as 'rf_hvac_forecast_best_complex.pkl'.")

# ========================================
# Feature Importances Visualization
# ========================================

# Note: With the pipeline, the RandomForest is preceded by PolynomialFeatures and PCA.
# The feature importances correspond to the PCA components, not the original features.
rf_model = best_model_final.named_steps['rf']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importances (PCA Components) - Random Forest")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xlabel("PCA Component Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
