import os
import json
from datetime import datetime, timezone
from time import sleep
import requests
from pysolar.solar import get_altitude, get_azimuth

# Define the location and constants
location = "Solbråveien 43, Asker, Norway"
latitude = 59.8338
longitude = 10.4376
altitude = 150  # Approximate altitude in meters
output_file = "yrWeatherData.json"
backup_dir = "yrBackups"

# Define the YR API endpoints and headers
yr_api_url_v2 = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
yr_api_url_v1 = "https://api.met.no/weatherapi/locationforecast/1.0/"
headers = {
    "User-Agent": "WeatherScript/1.0 (https://github.com/Prebuss)"
}

def fetch_weather(lat, lon, alt, use_v2=True):
    """
    Attempts to fetch data from Yr's newest API (v2).
    If that fails, falls back to older v1 endpoint.
    """
    url = yr_api_url_v2 if use_v2 else yr_api_url_v1
    params = {"lat": lat, "lon": lon, "altitude": alt} if use_v2 else {}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json(), response.headers
    except requests.exceptions.RequestException as e:
        print(f"Error fetching YR.no weather data ({'v2' if use_v2 else 'v1'}): {e}")
        return None, None

def calculate_solar_angles(lat, lon):
    """
    Returns current solar altitude and azimuth at the given lat/lon.
    """
    current_time = datetime.now(timezone.utc)
    solar_alt = get_altitude(lat, lon, current_time)
    solar_az = get_azimuth(lat, lon, current_time)
    return {
        "time": current_time.isoformat(),
        "solar_altitude": solar_alt,
        "solar_azimuth": solar_az,
    }

def save_full_json(yr_data):
    """
    Saves the full JSON payload from Yr into a timestamped file in backup_dir.
    """
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_filename = os.path.join(backup_dir, f"yr_{timestamp}.json")
    with open(backup_filename, "w", encoding="utf-8") as backup_file:
        json.dump(yr_data, backup_file, indent=4)
    print(f"Full YR data saved to {backup_filename}")

def merge_timeseries(existing_timeseries, new_timeseries):
    """
    Merges new_timeseries into existing_timeseries, removing duplicates by 'time'
    and returning a sorted list.
    """
    if not existing_timeseries:
        existing_timeseries = []
    if not new_timeseries:
        return existing_timeseries

    combined = existing_timeseries + new_timeseries
    timeseries_dict = {}

    # Each item is something like: {"time": "...", "data": {...}}
    for item in combined:
        # Overwrite by 'time' if duplicate
        timeseries_dict[item["time"]] = item

    merged_list = list(timeseries_dict.values())
    merged_list.sort(key=lambda x: x["time"])
    return merged_list

def update_main_data_file(snapshot):
    """
    Loads the main JSON file (output_file), appends the new snapshot,
    merges timeseries, and saves back to disk.
    """
    # 1. Load existing structure or create a new one
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            main_data = json.load(f)
    else:
        main_data = {
            "location": location,
            "snapshots": [],
            "unified_timeseries": []
        }
    
    # 2. Append the entire new snapshot (keeps a full copy of all data from that run)
    main_data["snapshots"].append(snapshot)

    # 3. Merge timeseries into a single unified_timeseries array
    #    - snapshot["yr_data"]["properties"]["timeseries"] is the new forecast entries
    if "properties" in snapshot["yr_data"] and "timeseries" in snapshot["yr_data"]["properties"]:
        new_ts = snapshot["yr_data"]["properties"]["timeseries"]
        main_data["unified_timeseries"] = merge_timeseries(main_data["unified_timeseries"], new_ts)

    # 4. Save updated JSON to disk
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(main_data, f, indent=4)
    print(f"Main JSON updated in {output_file}")

if __name__ == "__main__":
    print(f"Fetching weather data for {location} using YR.no API...")

    # Try fetching data from version 2.0
    yr_data, yr_headers = fetch_weather(latitude, longitude, altitude, use_v2=True)
    if not yr_data:
        print("Version 2.0 failed, falling back to version 1.0")
        yr_data, yr_headers = fetch_weather(latitude, longitude, altitude, use_v2=False)

    if yr_data:
        # 1. Save a complete backup of this run’s data
        save_full_json(yr_data)

        # 2. Calculate current solar angles
        solar_data = calculate_solar_angles(latitude, longitude)

        # 3. Build a snapshot object with all relevant info
        snapshot = {
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "location": location,
            "solar_data": solar_data,
            "expires": yr_headers.get("Expires") if yr_headers else None,
            "yr_data": yr_data
        }

        # 4. Update the main JSON file (append snapshot & unify timeseries)
        update_main_data_file(snapshot)

        # 5. Respect the 'Expires' header to avoid too frequent requests
        if yr_headers and yr_headers.get("Expires"):
            try:
                expires_time = datetime.strptime(
                    yr_headers["Expires"], "%a, %d %b %Y %H:%M:%S %Z"
                ).replace(tzinfo=timezone.utc)

                sleep_duration = (expires_time - datetime.now(timezone.utc)).total_seconds()
                if sleep_duration > 0:
                    print(f"Sleep for {int(sleep_duration)} seconds to respect API rate limits.")
                    # sleep(sleep_duration) # Sleep function for running script standalone.
            except ValueError:
                # If parsing fails, ignore sleeping
                pass
    else:
        print("Failed to fetch weather data.")

