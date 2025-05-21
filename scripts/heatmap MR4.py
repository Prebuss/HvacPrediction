import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###########################################
# 1) CONFIGURATION
###########################################
DATA_FOLDER = "/home/preben/Documents/Master/"
ADAX_FOLDER = os.path.join(DATA_FOLDER, "adax")
VENT_FOLDER = os.path.join(DATA_FOLDER, "vent")
QLARM_FOLDER= os.path.join(DATA_FOLDER, "qlarm")
YR_FOLDER   = os.path.join(DATA_FOLDER, "yr")

OUTPUT_CSV  = "merged_meetingroom_heatmap.csv"

# Filter string for Adax's meeting room
MEETING_ROOM_NAME = "Meeting room"  # or "Meeting room 4th floor", etc.

# In Qlarm, we pivot sensor data and keep only columns whose name includes "Meeting room"
MEETING_ROOM_Qlarm_SUBSTR = "Meeting room"

# If you want strict 5-min resampling, set True
RESAMPLE_5MIN = True

###########################################
# 2) MERGE ADAX (Meeting Room Only)
###########################################
def load_json_files(folder_path):
    """Loads all JSON files in a given folder as Python dicts."""
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

def merge_adax_meeting_room():
    """Merge only 'Meeting room' data from Adax. Sum energies if multiple devices present."""
    all_data = load_json_files(ADAX_FOLDER)
    dfs = []
    for data in all_data:
        temp_list = []
        for entry in data:
            timestamp = entry.get("timestamp")
            for room in entry.get("rooms", []):
                # Keep only the designated meeting room
                if room.get("name") and MEETING_ROOM_NAME in room["name"]:
                    energy_sum = sum(dev["energy"] for dev in room.get("devices", []))
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
        adax_df = pd.concat(dfs, ignore_index=True)
        adax_df.sort_values("timestamp", inplace=True)
        adax_df.reset_index(drop=True, inplace=True)
        return adax_df
    return pd.DataFrame()

###########################################
# 3) MERGE VENT (All)
###########################################
def merge_vent_all():
    vent_data = load_json_files(VENT_FOLDER)
    vent_dfs = []
    for data in vent_data:
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            vent_dfs.append(df)
    if vent_dfs:
        vent_df = pd.concat(vent_dfs, ignore_index=True)
        vent_df.sort_values("timestamp", inplace=True)
        vent_df.reset_index(drop=True, inplace=True)
        return vent_df
    return pd.DataFrame()

###########################################
# 4) MERGE YR (All)
###########################################
def merge_yr_all():
    yr_dfs = []
    if os.path.isdir(YR_FOLDER):
        for file in sorted(os.listdir(YR_FOLDER)):
            if file.endswith(".json"):
                file_path = os.path.join(YR_FOLDER, file)
                # e.g. "yr2025-03-09.json" -> "2025-03-09"
                date_str = file[2:12]
                with open(file_path, "r") as f:
                    data = json.load(f)
                df = pd.DataFrame([
                    {"timestamp": f"{date_str} {entry['time']}", **entry["data"]}
                    for entry in data
                ])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                yr_dfs.append(df)
    if yr_dfs:
        yr_df = pd.concat(yr_dfs, ignore_index=True)
        yr_df.sort_values("timestamp", inplace=True)
        yr_df.reset_index(drop=True, inplace=True)
        return yr_df
    return pd.DataFrame()

###########################################
# 5) MERGE Qlarm (Only Meeting Room Columns)
###########################################
def merge_qlarm_meeting_room():
    """
    Merge Qlarm, pivot, keep only columns whose name includes MEETING_ROOM_Qlarm_SUBSTR.
    """
    qlarm_dfs = []
    if not os.path.isdir(QLARM_FOLDER):
        return pd.DataFrame()
    for file in sorted(os.listdir(QLARM_FOLDER)):
        if file.endswith(".csv"):
            file_path = os.path.join(QLARM_FOLDER, file)
            df = pd.read_csv(file_path)
            if "Time" in df.columns:
                df.rename(columns={"Time": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # pivot
            df = df.pivot_table(
                index="timestamp",
                columns="SensorName",
                values="Value",
                aggfunc="mean"
            ).reset_index()
            qlarm_dfs.append(df)
    if qlarm_dfs:
        qlarm_df = pd.concat(qlarm_dfs, ignore_index=True)
        qlarm_df.sort_values("timestamp", inplace=True)
        qlarm_df.reset_index(drop=True, inplace=True)
        # Now filter columns
        # keep "timestamp" plus any column with "Meeting room" in the name
        keep_cols = ["timestamp"]
        for c in qlarm_df.columns:
            if c.lower().find(MEETING_ROOM_Qlarm_SUBSTR.lower()) != -1:
                keep_cols.append(c)
        qlarm_df = qlarm_df[keep_cols]
        return qlarm_df
    return pd.DataFrame()

###########################################
# 6) COMBINE & CORRELATION HEATMAP
###########################################
def main():
    # 1) Merge Adax (only meeting room)
    adax_df = merge_adax_meeting_room()
    print(f"Adax meeting room shape: {adax_df.shape}")

    # 2) Merge Vent (all)
    vent_df = merge_vent_all()
    print(f"Vent shape: {vent_df.shape}")

    # 3) Merge YR (all)
    yr_df   = merge_yr_all()
    print(f"YR shape: {yr_df.shape}")

    # 4) Merge Qlarm (only meeting room columns)
    qlarm_df= merge_qlarm_meeting_room()
    print(f"Qlarm meeting room shape: {qlarm_df.shape}")

    # 5) Combine
    merged_df = (
        adax_df.merge(vent_df, on="timestamp", how="outer")
               .merge(yr_df,   on="timestamp", how="outer")
               .merge(qlarm_df,on="timestamp", how="outer")
    )
    merged_df.sort_values("timestamp", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    # Forward fill
    merged_df.ffill(inplace=True)

    # (Optional) strict 5-min resampling
    if RESAMPLE_5MIN:
        merged_df.set_index("timestamp", inplace=True)
        merged_df = merged_df.resample("5T").ffill()
        merged_df.reset_index(inplace=True)
        merged_df.sort_values("timestamp", inplace=True)

    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Merged data saved to '{OUTPUT_CSV}' with shape={merged_df.shape}")

    # ========== CORRELATION HEATMAP ==========
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found. Exiting.")
        return

    corr_matrix = merged_df[numeric_cols].corr()

    print("\n=== Correlation Matrix (rounded) ===")
    print(corr_matrix.round(2))

    plt.figure(figsize=(10, 8))
    plt.matshow(corr_matrix, fignum=0, cmap='bwr')  # red-blue heatmap
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.colorbar()
    plt.title("Meeting Room Heatmap: Vent+YR (all), Adax+Qlarm (meeting only)", pad=20)
    plt.show()

if __name__ == "__main__":
    main()
