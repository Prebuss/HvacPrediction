import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Concatenate, SpatialDropout1D, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.config.set_visible_devices([], 'GPU')

###########################################
# 1) CONFIG
###########################################
DATA_FOLDER  = "/home/preben/Documents/Master/"
YR_FOLDER    = os.path.join(DATA_FOLDER, "yr")
VENT_FOLDER  = os.path.join(DATA_FOLDER, "vent")
ADAX_FOLDER  = os.path.join(DATA_FOLDER, "adax")
QLARM_FOLDER = os.path.join(DATA_FOLDER, "qlarm")

OUTPUT_CSV   = "merged_meetingroom_onlyEnergy_n2.csv"
MODEL_NAME   = "meetingroom_onlyEnergy_best_cpu_nr2.keras"
LOG_FILE     = "training_results.txt"

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

# Toggle: if True, skip KerasTuner and use fixed "best" hyperparameters.
USE_BEST_HPARAMS = True

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
TIME_FEATURES = ["hour_of_day", "day_of_week", "is_weekend"]

###########################################
# 3) MERGE FUNCTIONS
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
            with open(os.path.join(folder_path, file), "r") as f:
                data_list.append(json.load(f))
    return data_list


def merge_yr():
    """Merge all YR JSON files into a single DataFrame."""
    dfs = []
    if not os.path.isdir(YR_FOLDER):
        print(f"⚠ YR folder not found: {YR_FOLDER}")
        return pd.DataFrame()
    for file in sorted(os.listdir(YR_FOLDER)):
        if file.endswith(".json"):
            date_str = file[2:12]
            data = json.load(open(os.path.join(YR_FOLDER, file)))
            df = pd.DataFrame([
                {"timestamp": f"{date_str} {entry['time']}", **entry['data']}
                for entry in data
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            dfs.append(df)
    if dfs:
        out = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
        return out
    return pd.DataFrame()


def merge_vent():
    """Merge all Vent JSON files into one DataFrame."""
    dfs = []
    if not os.path.isdir(VENT_FOLDER):
        print(f"⚠ Vent folder not found: {VENT_FOLDER}")
        return pd.DataFrame()
    for data in load_json_files(VENT_FOLDER):
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return pd.DataFrame()


def merge_adax_meeting_energy():
    """Merge Adax meeting room energy consumption."""
    rows = []
    if not os.path.isdir(ADAX_FOLDER):
        print(f"⚠ Adax folder not found: {ADAX_FOLDER}")
        return pd.DataFrame()
    for data in load_json_files(ADAX_FOLDER):
        for entry in data:
            ts = entry.get('timestamp')
            if not ts:
                continue
            for room in entry.get('rooms', []):
                if room.get('name') == MEETING_ROOM_ADAX_NAME:
                    energy_sum = sum(d['energy'] for d in room.get('devices', []))
                    rows.append({'timestamp': ts, 'energy_consumption': energy_sum})
    if rows:
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)
    return pd.DataFrame()


def merge_qlarm_meeting_temp():
    """Merge Qlarm CSVs and pivot to meeting room temperature."""
    dfs = []
    if not os.path.isdir(QLARM_FOLDER):
        print(f"⚠ Qlarm folder not found: {QLARM_FOLDER}")
        return pd.DataFrame()
    for file in sorted(os.listdir(QLARM_FOLDER)):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(QLARM_FOLDER, file))
            if 'Time' in df.columns:
                df = df.rename(columns={'Time':'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            pivot = df.pivot_table(index='timestamp', columns='SensorName', values='Value', aggfunc='mean').reset_index()
            dfs.append(pivot)
    if dfs:
        out = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
        if MEETING_ROOM_QLARM_T_COL in out.columns:
            return out[['timestamp', MEETING_ROOM_QLARM_T_COL]]
        else:
            print(f"⚠ Target '{MEETING_ROOM_QLARM_T_COL}' not found in Qlarm pivot.")
    return pd.DataFrame()


def merge_all_and_save():
    """Merge all sources, forward-fill, optional resample, and save CSV."""
    yr_df = merge_yr()
    vent_df = merge_vent()
    adax_df = merge_adax_meeting_energy()
    qlarm_df = merge_qlarm_meeting_temp()

    merged_df = (adax_df.merge(vent_df, on='timestamp', how='outer')
                         .merge(yr_df, on='timestamp', how='outer')
                         .merge(qlarm_df, on='timestamp', how='outer'))
    merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
    merged_df.ffill(inplace=True)
    if RESAMPLE_5MIN:
        merged_df = merged_df.set_index('timestamp').resample('5T').ffill().reset_index()
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved merged data to '{OUTPUT_CSV}' with shape={merged_df.shape}")
    return merged_df

# 4) CREATE SEQUENCES
###########################################
def create_sequences_with_time(dataX, dataY, df_time_cols, seq_length):
    X_seq, X_static, y_seq = [], [], []
    for i in range(len(dataX) - seq_length):
        seq_x = dataX[i : i + seq_length]
        seq_y = dataY[i + seq_length]
        t_feats = df_time_cols[i + seq_length]
        if not np.isnan(seq_x).any() and not np.isnan(seq_y):
            X_seq.append(seq_x)
            X_static.append(t_feats)
            y_seq.append(seq_y)
    return np.array(X_seq, dtype=np.float32), np.array(X_static, dtype=np.float32), np.array(y_seq, dtype=np.float32)

###########################################
# 5) BUILD MODEL (with tuner)
###########################################
def build_sequence_model(hp):
    rnn_type = hp.Choice('rnn_type', values=['LSTM', 'GRU'], default='LSTM')
    num_rnn_layers = hp.Int('num_rnn_layers', 1, 2, default=1)
    units_1 = hp.Int('units_1', 32, 512, step=32, default=64)
    dropout_1 = hp.Float('dropout_1', 0.0, 0.5, step=0.1, default=0.2)
    rec_dropout_1 = hp.Float('rec_dropout_1', 0.0, 0.5, step=0.1, default=0.2)
    l2_reg = hp.Float('l2_reg', 1e-5, 1e-2, sampling='log', default=1e-4)
    use_spatial_dropout = hp.Boolean('use_spatial_dropout', default=False)
    use_layernorm = hp.Boolean('use_layernorm', default=False)

    time_series_input = Input(shape=(SEQ_LENGTH, len(BASE_FEATURES)), name="time_series_input")
    static_input = Input(shape=(len(TIME_FEATURES),), name="static_input")
    x = time_series_input
    if use_spatial_dropout:
        x = SpatialDropout1D(0.1)(x)

    # RNN stack
    if rnn_type == 'LSTM':
        if num_rnn_layers == 1:
            x = LSTM(units_1, dropout=dropout_1, recurrent_dropout=rec_dropout_1,
                     kernel_regularizer=regularizers.l2(l2_reg), return_sequences=False)(x)
        else:
            x = LSTM(units_1, dropout=dropout_1, recurrent_dropout=rec_dropout_1,
                     kernel_regularizer=regularizers.l2(l2_reg), return_sequences=True)(x)
            units_2 = hp.Int('units_2', 32, 128, step=32, default=64)
            dropout_2 = hp.Float('dropout_2', 0.0, 0.5, step=0.1, default=0.2)
            rec_dropout_2 = hp.Float('rec_dropout_2', 0.0, 0.5, step=0.1, default=0.2)
            x = LSTM(units_2, dropout=dropout_2, recurrent_dropout=rec_dropout_2,
                     kernel_regularizer=regularizers.l2(l2_reg), return_sequences=False)(x)
    else:
        if num_rnn_layers == 1:
            x = GRU(units_1, dropout=dropout_1, recurrent_dropout=rec_dropout_1,
                    kernel_regularizer=regularizers.l2(l2_reg), return_sequences=False)(x)
        else:
            x = GRU(units_1, dropout=dropout_1, recurrent_dropout=rec_dropout_1,
                    kernel_regularizer=regularizers.l2(l2_reg), return_sequences=True)(x)
            units_2 = hp.Int('units_2', 32, 128, step=32, default=64)
            dropout_2 = hp.Float('dropout_2', 0.0, 0.5, step=0.1, default=0.2)
            rec_dropout_2 = hp.Float('rec_dropout_2', 0.0, 0.5, step=0.1, default=0.2)
            x = GRU(units_2, dropout=dropout_2, recurrent_dropout=rec_dropout_2,
                    kernel_regularizer=regularizers.l2(l2_reg), return_sequences=False)(x)

    if use_layernorm:
        x = LayerNormalization()(x)

    # Static branch
    dense_units_static = hp.Int('static_dense_units', 8, 64, step=8, default=16)
    s = Dense(dense_units_static, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(static_input)
    s = Dense(dense_units_static // 2, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(s)

    merged = Concatenate()([x, s])
    final_dense_units = hp.Int('final_dense_units', 16, 64, step=16, default=32)
    out = Dense(final_dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(merged)
    out = Dense(1)(out)

    model = Model(inputs=[time_series_input, static_input], outputs=out)
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
                  loss='mae', metrics=['mae', 'mse'])
    return model

###########################################
# 6) FIXED "BEST" HYPERPARAMETERS
###########################################
def get_best_hyperparameters():
    return {
        'rnn_type': 'LSTM',
        'num_rnn_layers': 1,
        'units_1': 416,
        'dropout_1': 0.2,
        'rec_dropout_1': 0.0,
        'l2_reg': 1e-05,
        'use_spatial_dropout': True,
        'use_layernorm': False,
        'static_dense_units': 32,
        'final_dense_units': 16,
        'learning_rate': 0.0001,
        'units_2': 64,
        'dropout_2': 0.0,
        'rec_dropout_2': 0.0,
    }

class FixedHP:
    def __init__(self, params): self.params = params
    def Choice(self, name, values, default=None): return self.params[name]
    def Int(self, name, min_value, max_value, step=None, default=None): return self.params[name]
    def Float(self, name, min_value, max_value, step=None, sampling=None, default=None): return self.params[name]
    def Boolean(self, name, default=None): return self.params[name]

def build_sequence_model_fixed(params):
    hp = FixedHP(params)
    return build_sequence_model(hp)

###########################################
# 7) LOGGING FUNCTION
###########################################
def save_hparams_and_results(params, history, filepath=LOG_FILE):
    """
    Writes hyperparameters and final training/validation losses to a text file.
    """
    with open(filepath, 'w') as f:
        f.write('Hyperparameters used:\n')
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write('\nFinal Training Results:\n')
        for metric, values in history.history.items():
            f.write(f"{metric}: {values[-1]:.6f}\n")
    print(f"Saved hyperparameters and training results to {filepath}")

###########################################
# 8) EVALUATION FUNCTIONS
###########################################
def evaluate_model_unscaled(model, X_test_seq, X_test_static, y_test, scaler_y):
    """
    Evaluate the model on test data, inverse transform predictions, and plot results.
    """
    # Predict and inverse transform
    preds = model.predict([X_test_seq, X_test_static]).flatten()
    preds_unscaled = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Compute metrics
    mae  = mean_absolute_error(y_test_unscaled, preds_unscaled)
    mse  = mean_squared_error(y_test_unscaled, preds_unscaled)
    rmse = sqrt(mse)
    r2   = r2_score(y_test_unscaled, preds_unscaled)

    print("===== TEST SET PERFORMANCE (Unscaled) =====")
    print(f"MAE  = {mae:.4f}")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R^2  = {r2:.4f}")

    # Plot actual vs predicted over last 500 points
    n_points = min(500, len(y_test_unscaled))
    plt.figure()
    plt.plot(y_test_unscaled[-n_points:], label="Actual Temperature")
    plt.plot(preds_unscaled[-n_points:], label="Predicted Temperature")
    plt.title("Temperature Prediction: Last 500 Test Steps (Unscaled)")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()

    # Scatter plot predicted vs actual
    plt.figure()
    plt.scatter(y_test_unscaled, preds_unscaled, alpha=0.3)
    plt.title("Predicted vs. Actual Temperature (Unscaled)")
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.show()

    # Residual histogram
    residuals = y_test_unscaled - preds_unscaled
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution (Unscaled Temperature)")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Count")
    plt.show()

###########################################
# 8b) ADDITIONAL DIAGNOSTIC PLOTS
###########################################
def plot_learning_curves(history, metric='loss'):
    """train & validation curves (works for Keras History)."""
    plt.figure(figsize=(6,4))
    plt.plot(history.history[metric],     label=f'train_{metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.title(f'Learning curve – {metric}')
    plt.legend();  plt.tight_layout()
    plt.show()

def parity_and_residuals(model,
                         X_train_seq,  X_train_static,  y_train,
                         X_test_seq,   X_test_static,   y_test,
                         scaler_y):
    """
    Parity & residual-vs-fitted plots for train and test splits (un-scaled).
    """
    # ---------- predictions ----------
    y_pred_train = model.predict([X_train_seq, X_train_static]).flatten()
    y_pred_test  = model.predict([X_test_seq,  X_test_static ]).flatten()

    y_pred_train = scaler_y.inverse_transform(y_pred_train.reshape(-1,1)).ravel()
    y_pred_test  = scaler_y.inverse_transform(y_pred_test.reshape(-1,1)).ravel()
    y_train      = scaler_y.inverse_transform(y_train.reshape(-1,1)).ravel()
    y_test       = scaler_y.inverse_transform(y_test.reshape(-1,1)).ravel()

    # ---------- parity plot ----------
    lims = [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())]
    plt.figure(figsize=(5,5))
    plt.plot(lims, lims, 'k--', lw=1)
    plt.scatter(y_train, y_pred_train, s=8, alpha=0.4, label='train')
    plt.scatter(y_test,  y_pred_test,  s=8, alpha=0.4, label='test')
    plt.xlabel('Actual temperature')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual (train & test)')
    plt.legend();  plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # ---------- residual-vs-fitted ----------
    resid_train = y_pred_train - y_train
    resid_test  = y_pred_test  - y_test
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred_train, resid_train, s=8, alpha=0.4, label='train')
    plt.scatter(y_pred_test,  resid_test,  s=8, alpha=0.4, label='test')
    plt.axhline(0, color='k', lw=1)
    plt.xlabel('Predicted temperature')
    plt.ylabel('Residual (Pred − Actual)')
    plt.title('Residuals vs Fitted')
    plt.legend();  plt.tight_layout()
    plt.show()

###########################################
# 9) MAIN
###########################################
def main():
    merged_df = merge_all_and_save()
    if TARGET not in merged_df.columns:
        raise ValueError(f"Target '{TARGET}' not found in merged DataFrame columns.")

    merged_df['hour_of_day'] = merged_df['timestamp'].dt.hour
    merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
    merged_df['is_weekend']  = merged_df['day_of_week'].isin([5,6]).astype(int)

    used_df = merged_df[BASE_FEATURES + [TARGET] + TIME_FEATURES].copy()

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    used_df[BASE_FEATURES + TIME_FEATURES] = scaler_features.fit_transform(used_df[BASE_FEATURES + TIME_FEATURES])
    used_df[[TARGET]] = scaler_target.fit_transform(used_df[[TARGET]])

    X_seq, X_static, y_seq = create_sequences_with_time(
        used_df[BASE_FEATURES].values,
        used_df[TARGET].values,
        used_df[TIME_FEATURES].values,
        SEQ_LENGTH
    )

    n = len(X_seq)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    X_train_seq, X_val_seq, X_test_seq = X_seq[:train_end], X_seq[train_end:val_end], X_seq[val_end:]
    X_train_static, X_val_static, X_test_static = X_static[:train_end], X_static[train_end:val_end], X_static[val_end:]
    y_train, y_val, y_test = y_seq[:train_end], y_seq[train_end:val_end], y_seq[val_end:]

    callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)]

    if USE_BEST_HPARAMS:
        print("Using fixed hyperparameters, skipping tuner.")
        params = get_best_hyperparameters()
        model = build_sequence_model_fixed(params)
        print("\n===== Training model with fixed hyperparameters =====")
        history = model.fit(
            [X_train_seq, X_train_static], y_train,
            epochs=50,
            validation_data=([X_val_seq, X_val_static], y_val),
            callbacks=callbacks,
            verbose=1 
        )
        # ── extra diagnostics ─────────────────────────────────────────── #
        plot_learning_curves(history, metric='loss')          # loss curve
        plot_learning_curves(history, metric='mae')           # optional: mae curve
        save_hparams_and_results(params, history)
        best_model = model
    else:
        tuner = kt.BayesianOptimization(
            build_sequence_model,
            objective='val_loss',
            max_trials=10,
            num_initial_points=5,
            alpha=1e-4,
            beta=2.6,
            directory="meetingroom_onlyEnergy_bayes_dir",
            project_name="meetingroom_onlyEnergy_cpu",
            overwrite=True
        )
        print("\n===== Starting Bayesian Optimization (CPU mode) =====")
        tuner.search(
            [X_train_seq, X_train_static], y_train,
            epochs=50,
            validation_data=([X_val_seq, X_val_static], y_val),
            callbacks=callbacks,
            verbose=1
        )
        best_models = tuner.get_best_models(num_models=1)
        best_model = best_models[0] if best_models else None
        # ── optional learning-curve diagnostics for the tuner path ──────────────
        if best_model is not None:
            # 1) See if KerasTuner already attached a History object
            history = getattr(best_model, "history", None)

            # 2) If not, run a quick fine-tune to generate one
            if history is None:
                history = best_model.fit(
                    [X_train_seq, X_train_static], y_train,
                    epochs=50,                                 # adjust if you like
                    validation_data=([X_val_seq, X_val_static], y_val),
                    callbacks=callbacks,
                    verbose=1
                )

            # 3) Plot learning curves
            plot_learning_curves(history, "loss")
            plot_learning_curves(history, "mae")   # comment out if you only need loss
        else:
            print("⚠ No best model returned by tuner – skipping learning-curve plots.")


    if best_model:
        evaluate_model_unscaled(best_model, X_test_seq, X_test_static, y_test, scaler_target)
        best_model.save(MODEL_NAME)
        print(f"✅ Model saved as '{MODEL_NAME}'.")
        parity_and_residuals(best_model,
            X_train_seq,  X_train_static,  y_train,
            X_test_seq,   X_test_static,   y_test,
            scaler_target)
    else:
        print("No model available to evaluate.")
    


if __name__ == "__main__":
    main()
