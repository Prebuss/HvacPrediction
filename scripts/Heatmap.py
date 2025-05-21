#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

# =====================
# Data Loading Functions
# =====================

def load_json_files(folder_path):
    """Load and return a list of JSON data from a folder."""
    data_list = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                data_list.append(json.load(f))
    return data_list

def load_yr_data(yr_folder):
    """Process YR Weather Forecast Data from JSON files."""
    yr_dfs = []
    for file in sorted(os.listdir(yr_folder)):
        if file.endswith(".json"):
            file_path = os.path.join(yr_folder, file)
            # Extract date from filename; adjust slicing if needed
            date_str = file[2:12]  # e.g., "yr2025-03-09.json" -> "2025-03-09"
            with open(file_path, "r") as f:
                data = json.load(f)
            df = pd.DataFrame([
                {"timestamp": f"{date_str} {entry['time']}", **entry["data"]}
                for entry in data
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            yr_dfs.append(df)
    return pd.concat(yr_dfs, ignore_index=True) if yr_dfs else pd.DataFrame()

def load_vent_data(vent_folder):
    """Process Ventilation Data from JSON files."""
    vent_data = load_json_files(vent_folder)
    vent_dfs = []
    for data in vent_data:
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        vent_dfs.append(df)
    return pd.concat(vent_dfs, ignore_index=True) if vent_dfs else pd.DataFrame()

def load_adax_data(adax_folder):
    """Process Adax Heating Panel Data from JSON files."""
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
    return pd.concat(adax_dfs, ignore_index=True) if adax_dfs else pd.DataFrame()

def load_qlarm_data(qlarm_folder):
    """Process Qlarm Data from CSV files."""
    qlarm_dfs = []
    for file in os.listdir(qlarm_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(qlarm_folder, file)
            df = pd.read_csv(file_path)
            df.rename(columns={"Time": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Pivot in case of duplicate sensor entries
            df = df.pivot_table(index="timestamp", columns="SensorName", values="Value", aggfunc="mean").reset_index()
            qlarm_dfs.append(df)
    return pd.concat(qlarm_dfs, ignore_index=True) if qlarm_dfs else pd.DataFrame()

def merge_datasets(data_folder):
    """Load all datasets and merge them on timestamp."""
    # Define folder paths (adjust as needed)
    yr_folder = os.path.join(data_folder, "yr")
    vent_folder = os.path.join(data_folder, "vent")
    adax_folder = os.path.join(data_folder, "adax")
    qlarm_folder = os.path.join(data_folder, "qlarm")
    
    # Load each dataset
    yr_df = load_yr_data(yr_folder)
    vent_df = load_vent_data(vent_folder)
    adax_df = load_adax_data(adax_folder)
    qlarm_df = load_qlarm_data(qlarm_folder)
    
    # Merge datasets on timestamp (outer join)
    merged_df = (
        adax_df.merge(vent_df, on="timestamp", how="outer")
               .merge(yr_df, on="timestamp", how="outer")
               .merge(qlarm_df, on="timestamp", how="outer")
    )
    # Sort and fill missing values
    merged_df.sort_values(by="timestamp", inplace=True)
    merged_df.ffill(inplace=True)
    
    return merged_df

# =====================
# Analysis Functions
# =====================

def plot_correlation_heatmap(df, features, output_file="correlation_heatmap.png"):
    """
    Calculate and plot the correlation heatmap for the specified features.
    Only numeric columns are considered.
    """
    numeric_df = df[features].select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    # Use a diverging colormap with fixed vmin and vmax to ensure -1 is blue and 1 is red
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"Correlation heatmap saved as {output_file}")
    return corr


def perform_variance_threshold(df, features, threshold=0.1):
    """
    Remove features with low variance from the specified list.
    Only numeric features are tested.
    """
    numeric_df = df[features].select_dtypes(include=[np.number])
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric_df)
    features_kept = numeric_df.columns[selector.get_support()]
    removed_features = set(numeric_df.columns) - set(features_kept)
    print("Features kept after variance thresholding:", list(features_kept))
    if removed_features:
        print("Features removed due to low variance:", list(removed_features))
    else:
        print("No features were removed due to low variance.")
    return numeric_df[features_kept]

def plot_pca_variance(numeric_df, output_file="pca_explained_variance.png"):
    """Perform PCA and plot the cumulative explained variance ratio."""
    # Use the smaller of number of features or samples to define n_components
    n_components = min(numeric_df.shape[1], numeric_df.shape[0])
    pca = PCA(n_components=n_components)
    # Drop missing values before PCA
    numeric_df = numeric_df.dropna()
    pca.fit(numeric_df)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA - Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"PCA explained variance plot saved as {output_file}")
    print("Cumulative explained variance by component:", explained_variance)

# =====================
# Main Function
# =====================

def main():
    parser = argparse.ArgumentParser(
        description="Merge HVAC datasets, generate a correlation heatmap, perform variance threshold testing, and run PCA."
    )
    parser.add_argument("--data_folder", type=str, default="/home/preben/Documents/Master/",
                        help="Path to the main data folder (default: /home/preben/Documents/Master/)")
    parser.add_argument("--var_threshold", type=float, default=0.1,
                        help="Variance threshold for feature selection (default: 0.1)")
    args = parser.parse_args()
    
    print("🔄 Loading and merging HVAC data...")
    merged_df = merge_datasets(args.data_folder)
    print(f"✅ Merged dataset contains {len(merged_df)} entries and {len(merged_df.columns)} columns.")
    
    # Optionally save the merged dataset
    merged_df.to_csv("merged_hvac_dataset.csv", index=False)
    print("✅ Merged dataset saved as 'merged_hvac_dataset.csv'.")
    
    # Define training features (as used in your training script)
    features = [
        "target_temp", "current_temp", "energy_consumption",
        "co2", "supplyAirflow", "extractAirflow", "supplyAirDuctPressure",
        "extractAirDuctPressure", "supplyAirFanSpeedLevel", "extractAirFanSpeedLevel",
        "supplyAirTemperature", "extractAirTemperature", "outdoorTemperature",
        "reheatLevel", "coolingLevel", "heatExchangerRegulator", "RhxEfficiency",
        "air_temperature", "relative_humidity", "wind_speed"
    ]
    
    # Check that these features exist in the merged dataset.
    available_features = [f for f in features if f in merged_df.columns]
    if len(available_features) != len(features):
        missing = set(features) - set(available_features)
        print("Warning: The following features are missing in the merged dataset:", missing)
    
    # Print summary and missing values for the selected features
    print("\nData Summary for Selected Features:")
    print(merged_df[available_features].describe())
    missing_vals = merged_df[available_features].isnull().mean() * 100
    print("\nPercentage of missing values per feature:")
    print(missing_vals)
    
    # Optionally, normalize the features (as in your training script)
    scaler = MinMaxScaler()
    merged_df[available_features] = scaler.fit_transform(merged_df[available_features])
    
    # Generate Correlation Heatmap
    print("\nGenerating correlation heatmap...")
    corr = plot_correlation_heatmap(merged_df, available_features)
    
    # Perform Variance Threshold Test
    print("\nPerforming variance threshold test...")
    vt_df = perform_variance_threshold(merged_df, available_features, threshold=args.var_threshold)
    
    # Run PCA on the numeric features after variance thresholding
    print("\nRunning PCA on the remaining numeric features...")
    plot_pca_variance(vt_df)
    
if __name__ == "__main__":
    main()
