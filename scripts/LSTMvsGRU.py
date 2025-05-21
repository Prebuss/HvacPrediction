import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ✅ Load Trained Models
lstm_model = tf.keras.models.load_model("lstm_hvac_forecast.keras")
rf_model = joblib.load("rf_hvac_forecast.pkl")
print("✅ Loaded LSTM and Random Forest models.")

# ✅ Load New Data
new_data_path = "new_hvac_data.csv"  # Update path if needed
new_df = pd.read_csv(new_data_path)

# ✅ Ensure timestamp is a datetime object
new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])

# ✅ Select Features
features = [
    "target_temp", "current_temp", "energy_consumption",
    "co2", "supplyAirflow", "extractAirflow", "supplyAirDuctPressure",
    "extractAirDuctPressure", "supplyAirFanSpeedLevel", "extractAirFanSpeedLevel",
    "supplyAirTemperature", "extractAirTemperature", "outdoorTemperature",
    "reheatLevel", "coolingLevel", "heatExchangerRegulator", "RhxEfficiency",
    "air_temperature", "relative_humidity", "wind_speed"
]
target = "current_temp"

# ✅ Handle Missing Data
if new_df.isnull().sum().sum() > 0:
    print("⚠️ New data contains NaN values! Filling missing values...")
    new_df.ffill(inplace=True)

# ✅ Normalize Data (Required for LSTM)
scaler = MinMaxScaler()
new_df[features] = scaler.fit_transform(new_df[features])

# ✅ Define Sequence Length (LSTM)
SEQ_LENGTH = 24

# ✅ Create Test Sequences
def create_test_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq_X = data[i:i+seq_length]
        seq_y = data[i+seq_length][features.index(target)]

        if np.isnan(seq_X).sum() == 0 and not np.isnan(seq_y):
            X.append(seq_X)
            y.append(seq_y)

    return np.array(X), np.array(y)

X_test_lstm, y_test_lstm = create_test_sequences(new_df[features].values, SEQ_LENGTH)
X_test_rf = new_df[features].values[:-1]  # Shift target for RF alignment
y_test_rf = new_df[target].values[1:]  # Shift target for RF alignment

# ✅ Make Predictions
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_rf = rf_model.predict(X_test_rf)

# ✅ Convert Predictions Back to Original Scale
y_pred_lstm_actual = scaler.inverse_transform(y_pred_lstm)
y_pred_rf_actual = scaler.inverse_transform(y_pred_rf.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test_rf.reshape(-1, 1))

# ✅ Evaluate Performance
def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 {model_name} Performance:")
    print(f"🔹 MSE: {mse:.4f}")
    print(f"🔹 RMSE: {rmse:.4f}")
    print(f"🔹 MAE: {mae:.4f}")
    print(f"🔹 R² Score: {r2:.4f}")

    return mse, rmse, mae, r2

mse_lstm, rmse_lstm, mae_lstm, r2_lstm = evaluate_model(y_test_actual, y_pred_lstm_actual, "LSTM")
mse_rf, rmse_rf, mae_rf, r2_rf = evaluate_model(y_test_actual, y_pred_rf_actual, "Random Forest")

# ✅ Compare Model Performance Visually
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label="Actual Temperature", color="blue")
plt.plot(y_pred_lstm_actual, label="LSTM Prediction", color="red", linestyle="dashed")
plt.plot(y_pred_rf_actual, label="RF Prediction", color="green", linestyle="dashed")
plt.xlabel("Time Steps")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.title("LSTM vs Random Forest Predictions vs Actual")
plt.show()

# ✅ Plot Error Distribution
errors_lstm = y_test_actual - y_pred_lstm_actual
errors_rf = y_test_actual - y_pred_rf_actual

plt.figure(figsize=(10, 5))
plt.hist(errors_lstm, bins=30, alpha=0.7, label="LSTM", color="red", edgecolor="black")
plt.hist(errors_rf, bins=30, alpha=0.7, label="Random Forest", color="green", edgecolor="black")
plt.xlabel("Prediction Error (°C)")
plt.ylabel("Frequency")
plt.title("Error Distribution: LSTM vs Random Forest")
plt.legend()
plt.show()

# ✅ Compare Performance in a Bar Chart
import seaborn as sns

metrics = ["MSE", "RMSE", "MAE", "R²"]
values_lstm = [mse_lstm, rmse_lstm, mae_lstm, r2_lstm]
values_rf = [mse_rf, rmse_rf, mae_rf, r2_rf]

df_comparison = pd.DataFrame({"Metric": metrics, "LSTM": values_lstm, "Random Forest": values_rf})
df_comparison = df_comparison.melt(id_vars="Metric", var_name="Model", value_name="Score")

plt.figure(figsize=(8, 5))
sns.barplot(x="Metric", y="Score", hue="Model", data=df_comparison)
plt.title("LSTM vs Random Forest: Performance Comparison")
plt.ylabel("Score")
plt.show()
