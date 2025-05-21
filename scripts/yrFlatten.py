#!/usr/bin/env python3

import json
import os
import glob
from datetime import datetime, timedelta, timezone

def flatten_first_full_day(input_path, output_path):
    """
    Reads each JSON file in `input_path`, looks for the earliest timestamp
    that is exactly 00:00:00 UTC, and collects data for the next 24 hours
    (from that start 00:00:00 to the next day's 00:00:00, exclusive).
    Flattens the nested 'details' fields and writes the result to a file 
    named yrYYYY-MM-DD.json (based on that first midnight's date) in `output_path`.
    """

    os.makedirs(output_path, exist_ok=True)

    for file_name in glob.glob(os.path.join(input_path, "*.json")):
        with open(file_name, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Pull out timeseries
        timeseries = raw_data.get("properties", {}).get("timeseries", [])
        if not timeseries:
            print(f"No timeseries found in {file_name}, skipping.")
            continue

        # 1) Parse and sort all timestamps so we can find the first 00:00:00
        parsed_times = []  # will hold tuples: (dt, original_item)
        for item in timeseries:
            t_str = item["time"]  # e.g. "2025-02-17T23:00:00Z"
            dt = datetime.fromisoformat(t_str.replace("Z", "+00:00"))  # offset-aware in UTC
            parsed_times.append((dt, item))

        # Sort by the datetime so we can iterate in ascending order
        parsed_times.sort(key=lambda x: x[0])

        # 2) Find the first occurrence of exactly hour=0, minute=0, second=0
        start_dt = None
        for dt, _ in parsed_times:
            if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                start_dt = dt
                break

        if not start_dt:
            # If there's no exact 00:00:00 in the file, skip it or handle as desired
            print(f"No 00:00:00 found in {file_name}; skipping.")
            continue

        # 3) Define the end_dt as start_dt + 24 hours
        end_dt = start_dt + timedelta(days=1)

        # 4) Filter for entries where start_dt <= dt < end_dt
        #    That is all times from that midnight up to (but not including) next midnight
        filtered_entries = []
        for dt, item in parsed_times:
            if start_dt <= dt < end_dt:
                filtered_entries.append(item)

        if not filtered_entries:
            print(f"No entries from {start_dt} to {end_dt} in {file_name}, skipping.")
            continue

        # 5) Flatten each filtered entry:
        #    - "time": use HH:MM:SS
        #    - "data": merge the details from 'instant', 'next_1_hours', 'next_6_hours', 'next_12_hours'
        output_records = []
        for item in filtered_entries:
            dt_str = item["time"]
            dt_obj = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            time_str = dt_obj.strftime("%H:%M:%S")  # "HH:MM:SS"

            entry_data = item["data"]
            flattened = dict(entry_data.get("instant", {}).get("details", {}))

            for forecast_block in ("next_1_hours", "next_6_hours", "next_12_hours"):
                if forecast_block in entry_data:
                    flattened.update(entry_data[forecast_block].get("details", {}))

            output_records.append({
                "time": time_str,
                "data": flattened
            })

        # 6) Use start_dt’s date to name the output file
        date_str = start_dt.strftime("%Y-%m-%d")
        out_file = os.path.join(output_path, f"yr{date_str}.json")
        with open(out_file, "w", encoding="utf-8") as out_f:
            json.dump(output_records, out_f, indent=4)

        print(f"Flattened {file_name} from {start_dt} to {end_dt}, total {len(output_records)} entries -> {out_file}.")


if __name__ == "__main__":
    INPUT_DIR = "/tmp/Master/yrflat"       # Where original JSON files live
    OUTPUT_DIR = "/tmp/Master/yr" # Where to place flattened results

    flatten_first_full_day(INPUT_DIR, OUTPUT_DIR)
