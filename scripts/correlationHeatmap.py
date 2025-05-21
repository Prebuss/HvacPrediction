import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###########################################
# 1) CONFIGURATION
###########################################
DATA_FOLDER = "/home/preben/Documents/Master/"

# Folders for "val" data
VAL_ADAX_FOLDER = os.path.join(DATA_FOLDER, "valAdax")
VAL_QLARM_FOLDER= os.path.join(DATA_FOLDER, "valQlarm")
VAL_VENT_FOLDER = os.path.join(DATA_FOLDER, "valVent")
VAL_YR_FOLDER   = os.path.join(DATA_FOLDER, "valYr")

# Folders for "regular" data
ADAX_FOLDER = os.path.join(DATA_FOLDER, "adax")
QLARM_FOLDER= os.path.join(DATA_FOLDER, "qlarm")
VENT_FOLDER = os.path.join(DATA_FOLDER, "vent")
YR_FOLDER   = os.path.join(DATA_FOLDER, "yr")

OUTPUT_CSV  = "merged_all_data_corr.csv"

###########################################
# 2) HELPER FUNCTIONS
###########################################
def load_json_files(folder_path):
    """Loads all JSON files in a given folder, returns list of dict objects."""
    data_list = []
    if not os.path.isdir(folder_path):
        print(f"⚠ Folder not found: {folder_path}")
        return data_list
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                data_list.append(json.load(f))
    return data_list

def merge_adax(folder_path):
    dfs = []
    adax_data = load_json_files(folder_path)
    for data in adax_data:
        temp_list = []
        for entry in data:
            timestamp = entry.get("timestamp")
            for room in entry.get("rooms", []):
                energy_sum = sum(d["energy"] for d in room.get("devices", []))
                temp_list.append({
                    "timestamp": timestamp,
                    "room_id": room.get("id"),
                    "room_name": room.get("name"),
                    "target_temp": room.get("targetTemperature"),
                    "current_temp": room.get("currentTemperature"),
                    "energy_consumption": energy_sum
                })
        df = pd.DataFrame(temp_list)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            dfs.append(df)
    if dfs:
        out = pd.concat(dfs, ignore_index=True)
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out
    return pd.DataFrame()

def merge_vent(folder_path):
    dfs = []
    vent_data = load_json_files(folder_path)
    for data in vent_data:
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            dfs.append(df)
    if dfs:
        out = pd.concat(dfs, ignore_index=True)
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out
    return pd.DataFrame()

def merge_yr(folder_path):
    dfs = []
    if os.path.isdir(folder_path):
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".json"):
                file_path = os.path.join(folder_path, file)
                # e.g. "yr2025-03-09.json" -> date_str="2025-03-09"
                date_str = file[2:12]
                with open(file_path, "r") as f:
                    data = json.load(f)
                df = pd.DataFrame([
                    {"timestamp": f"{date_str} {entry['time']}", **entry["data"]}
                    for entry in data
                ])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                dfs.append(df)
    if dfs:
        out = pd.concat(dfs, ignore_index=True)
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out
    return pd.DataFrame()

def merge_qlarm(folder_path):
    dfs = []
    if not os.path.isdir(folder_path):
        return pd.DataFrame()
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            if "Time" in df.columns:
                df.rename(columns={"Time": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # pivot sensors
            df = df.pivot_table(
                index="timestamp",
                columns="SensorName",
                values="Value",
                aggfunc="mean"
            ).reset_index()
            dfs.append(df)
    if dfs:
        out = pd.concat(dfs, ignore_index=True)
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out
    return pd.DataFrame()

###########################################
# 3) MERGE BOTH "val" + "regular" DATA
###########################################
def merge_all_data():
    """Merge Adax, Vent, YR, Qlarm from val + regular folders into one DataFrame."""
    print("Merging Adax (val + regular) ...")
    val_adax_df = merge_adax(VAL_ADAX_FOLDER)
    adax_df     = merge_adax(ADAX_FOLDER)
    full_adax   = pd.concat([val_adax_df, adax_df], ignore_index=True) if not val_adax_df.empty or not adax_df.empty else pd.DataFrame()

    print("Merging Vent (val + regular) ...")
    val_vent_df = merge_vent(VAL_VENT_FOLDER)
    vent_df     = merge_vent(VENT_FOLDER)
    full_vent   = pd.concat([val_vent_df, vent_df], ignore_index=True) if not val_vent_df.empty or not vent_df.empty else pd.DataFrame()

    print("Merging YR (val + regular) ...")
    val_yr_df = merge_yr(VAL_YR_FOLDER)
    yr_df     = merge_yr(YR_FOLDER)
    full_yr   = pd.concat([val_yr_df, yr_df], ignore_index=True) if not val_yr_df.empty or not yr_df.empty else pd.DataFrame()

    print("Merging Qlarm (val + regular) ...")
    val_qlarm_df = merge_qlarm(VAL_QLARM_FOLDER)
    qlarm_df     = merge_qlarm(QLARM_FOLDER)
    full_qlarm   = pd.concat([val_qlarm_df, qlarm_df], ignore_index=True) if not val_qlarm_df.empty or not qlarm_df.empty else pd.DataFrame()

    merged_df = (
        full_adax.merge(full_vent, on="timestamp", how="outer")
                 .merge(full_yr,   on="timestamp", how="outer")
                 .merge(full_qlarm,on="timestamp", how="outer")
    )
    merged_df.sort_values("timestamp", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    # Forward-fill missing data
    merged_df.ffill(inplace=True)

    # Combine ALL Qlarm temperature columns into a single average
    # (We'll call it "All Qlarm Average Temperature")
    # Then drop the original Qlarm temperature columns.
    qlarm_temp_cols = [
        c for c in merged_df.columns
        if ("Temperature" in c) and (c != "All Qlarm Average Temperature")
    ]
    if qlarm_temp_cols:
        merged_df["All Qlarm Average Temperature"] = merged_df[qlarm_temp_cols].mean(axis=1)
        # Remove the original Qlarm temperature columns
        merged_df.drop(columns=qlarm_temp_cols, inplace=True)

    # Save
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ All data merged. Shape={merged_df.shape}. Saved to '{OUTPUT_CSV}'.")

    return merged_df

###########################################
# 4) PLOT CORRELATION HEATMAP (RED-BLUE)
###########################################
def plot_correlation_heatmap(df):
    """
    Compute correlation among all numeric columns,
    plot it with a colormap from blue (negative corr) to red (positive corr).
    """
    import matplotlib.pyplot as plt

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found for correlation.")
        return

    corr_matrix = df[numeric_cols].corr()

    print("\nCorrelation Matrix (rounded):")
    print(corr_matrix.round(2))

    plt.figure()
    # 'bwr' = Blue-White-Red: negative = blue, positive = red, 0 ~ white
    plt.matshow(corr_matrix, fignum=0, cmap='bwr')  
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.colorbar()
    plt.title("Correlation Heatmap (Blue to Red)", pad=20)
    plt.show()

###########################################
# 5) MAIN
###########################################
def main():
    merged_df = merge_all_data()
    plot_correlation_heatmap(merged_df)

if __name__ == "__main__":
    main()
