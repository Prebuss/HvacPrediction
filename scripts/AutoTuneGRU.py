import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras_tuner as kt

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
folders = ["yr", "vent", "adax", "qlarm"]

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

# ✅ Define Sequence Length
SEQ_LENGTH = 24

# ✅ Create Sequences Function
def create_sequences(data, target_column, seq_length):
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

# ✅ Train-Test Split (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ======= AUTOMATIC HYPERPARAMETER TUNING ======= #
def build_model(hp):
    model = Sequential()
    
    # Tune the number of GRU layers (1 or 2)
    num_gru_layers = hp.Int('num_gru_layers', min_value=1, max_value=2, default=2)
    
    # --- First GRU Layer ---
    units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32, default=128)
    dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
    rec_dropout_1 = hp.Float('rec_dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
    
    if num_gru_layers > 1:
        # When using two GRU layers, the first must return sequences.
        model.add(GRU(units_1,
                      return_sequences=True,
                      input_shape=(SEQ_LENGTH, len(features)),
                      dropout=dropout_1,
                      recurrent_dropout=rec_dropout_1))
        
        # --- Second GRU Layer ---
        units_2 = hp.Int('units_2', min_value=32, max_value=128, step=16, default=64)
        dropout_2 = hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
        rec_dropout_2 = hp.Float('rec_dropout_2', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
        model.add(GRU(units_2,
                      dropout=dropout_2,
                      recurrent_dropout=rec_dropout_2))
    else:
        model.add(GRU(units_1,
                      input_shape=(SEQ_LENGTH, len(features)),
                      dropout=dropout_1,
                      recurrent_dropout=rec_dropout_1))
    
    # --- Dense Layers ---
    dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16, default=32)
    l2_reg = hp.Float('l2_reg', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
    model.add(Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)))
    
    # Final output layer (predicting a single continuous value)
    model.add(Dense(1))
    
    # --- Compile Model ---
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse")
    
    return model

# Set up the Keras Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,              # Number of hyperparameter combinations to try
    executions_per_trial=1,
    directory='hvac_tuner_dir',
    project_name='gru_hvac_tuning'
)

# Callbacks for early stopping and learning rate reduction
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

print("🔍 Starting hyperparameter search...")
tuner.search(X_train, y_train,
             epochs=50,
             validation_data=(X_test, y_test),
             callbacks=[early_stopping, reduce_lr],
             verbose=1)

# Retrieve the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("✅ Best hyperparameters found:")
print(best_hp.values)

# Evaluate the best model on the test set
test_loss = best_model.evaluate(X_test, y_test)
print(f"📉 Test Loss (MSE): {test_loss}")

# Save the best model
best_model.save("gru_hvac_forecast_best.keras")
print("✅ Best model saved as 'gru_hvac_forecast_best.keras'.")

# Optionally, plot the training loss of the best model (if available)
if 'history' in dir(best_model):
    history = best_model.history.history
    plt.figure(figsize=(10, 5))
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.title("GRU HVAC Training Loss Over Epochs")
    plt.show()
