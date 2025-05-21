import os
import glob
import csv

SENSORS = [
    {"name": "Preben - Canteen Humidity",            "tagId": "mac=AC233FA9867C;type=humidity"},
    {"name": "Preben - Canteen Temperature",         "tagId": "mac=AC233FA9867C;type=temperature"},
    {"name": "Preben - Meeting Room 434 Temperature","tagId": "mac=AC233FA98686;type=temperature"},
    {"name": "Preben - Office 401 - Humidity",       "tagId": "mac=AC233FA9869B;type=humidity"},
    {"name": "Preben - Office 401 - Temperature",    "tagId": "mac=AC233FA9869B;type=temperature"},
    {"name": "Preben - Office 402 - Humidity",       "tagId": "mac=AC233FA9869C;type=humidity"},
    {"name": "Preben - Office 402 - Temperature",    "tagId": "mac=AC233FA9869C;type=temperature"},
    {"name": "Preben - Office 403 - Humidity",       "tagId": "mac=AC233FA98654;type=humidity"},
    {"name": "Preben - Office 403 - Temperature",    "tagId": "mac=AC233FA98654;type=temperature"},
    {"name": "Preben - Office 404 - Humidity",       "tagId": "mac=AC233FA986EA;type=humidity"},
    {"name": "Preben - Office 404 - Temperature",    "tagId": "mac=AC233FA986EA;type=temperature"},
    {"name": "Preben - Office 405 - Humidity",       "tagId": "mac=AC233FA98656;type=humidity"},
    {"name": "Preben - Office 405 - Temperature",    "tagId": "mac=AC233FA98656;type=temperature"},
    {"name": "Preben - Office 406 - Humidity",       "tagId": "mac=AC233FA98655;type=humidity"},
    {"name": "Preben - Office 406 - Temperature",    "tagId": "mac=AC233FA98655;type=temperature"},
    {"name": "Preben - Office 407 - Humidity",       "tagId": "mac=AC233FA98675;type=humidity"},
    {"name": "Preben - Office 407 - Temperature",    "tagId": "mac=AC233FA98675;type=temperature"},
    {"name": "Preben - Office 408 - Humidity",       "tagId": "mac=AC233FA98680;type=humidity"},
    {"name": "Preben - Office 408 - Temperature",    "tagId": "mac=AC233FA98680;type=temperature"},
    {"name": "Preben - Office 409 - Humidity",       "tagId": "mac=AC233FA98689;type=humidity"},
    {"name": "Preben - Office 409 - Temperature",    "tagId": "mac=AC233FA98689;type=temperature"}
]

def merge_csv_to_daily_csv(input_folder):
    """
    For each day (10..24), gather CSV data from all sensors and write a single
    'qlarmData 2025.03.dd.csv' file with columns: SensorName, Time, Value.
    """
    # Ensure the output directory exists (create it if needed).
    os.makedirs(input_folder, exist_ok=True)

    for day in range(10, 25):
        # Collect rows from all sensors for this day
        aggregated_rows = []
        
        for sensor in SENSORS:
            sensor_name = sensor["name"]
            # We expect files named "<SensorName> 2025.03.dd.csv"
            file_pattern = os.path.join(
                input_folder,
                f"{sensor_name} 2025.03.{day:02d}.csv"
            )
            
            matched_files = glob.glob(file_pattern)
            
            for csv_file_path in matched_files:
                with open(csv_file_path, "r", encoding="utf-8") as csv_file:
                    lines = csv_file.read().splitlines()

                    # If the first line starts with "Tag name:", skip it
                    if lines and lines[0].startswith("Tag name:"):
                        csv_data = lines[1:]
                    else:
                        csv_data = lines

                    # We need at least a header line + 1 data row to parse
                    if len(csv_data) > 1:
                        reader = csv.DictReader(csv_data)  # expects columns "Time" and "Value"
                        for row in reader:
                            # Remove extra whitespace from keys
                            row = {key.strip(): value for key, value in row.items()}
                            # Add which sensor these readings belong to
                            row["SensorName"] = sensor_name
                            aggregated_rows.append(row)

        # Write out the aggregated CSV for this day
        out_filename = f"qlarmData 2025.03.{day:02d}.csv"
        out_path = os.path.join(input_folder, out_filename)

        fieldnames = ["SensorName", "Time", "Value"]

        with open(out_path, "w", newline="", encoding="utf-8") as out_csv:
            writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregated_rows)

        print(f"[Day {day:02d}] Wrote CSV with {len(aggregated_rows)} rows: {out_path}")

if __name__ == "__main__":
    input_folder = "/home/preben/Documents/Master/valQlarm"
    merge_csv_to_daily_csv(input_folder)
