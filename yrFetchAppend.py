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
output_file = "yr_weather_data.json"
backup_dir = "yr_backups"

# Define the YR API endpoints and headers
yr_api_url_v2 = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
yr_api_url_v1 = "https://api.met.no/weatherapi/locationforecast/1.0/"
headers = {
    "User-Agent": "WeatherScript/1.0 (https://github.com/yourusername/WeatherScript)"  # Replace with your project or script URL
}

# Function to fetch weather data from the YR API
def fetch_weather(lat, lon, alt, use_v2=True):
    url = yr_api_url_v2 if use_v2 else yr_api_url_v1
    params = {"lat": lat, "lon": lon, "altitude": alt} if use_v2 else {}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json(), response.headers
    except requests.exceptions.RequestException as e:
        print(f"Error fetching YR.no weather data ({'v2' if use_v2 else 'v1'}): {e}")
        return None, None

# Function to calculate solar angles
def calculate_solar_angles(lat, lon):
    current_time = datetime.now(timezone.utc)
    altitude = get_altitude(lat, lon, current_time)
    azimuth = get_azimuth(lat, lon, current_time)
    return {
        "time": current_time.isoformat(),
        "solar_altitude": altitude,
        "solar_azimuth": azimuth,
    }

# Function to save or append timeseries data to a JSON file
def save_or_append_json(new_data, filename):
    # Extract the relevant parts of the data (timeseries)
    new_timeseries = new_data["yr_data"]["properties"]["timeseries"]

    if os.path.exists(filename):
        try:
            # Read the existing file
            with open(filename, "r") as file:
                existing_data = json.load(file)

            # Append only the timeseries data
            if isinstance(existing_data, list):
                for entry in new_timeseries:
                    existing_data.append(entry)
            else:
                # If the file contains non-list data, create a list
                existing_data = new_timeseries
        except (json.JSONDecodeError, IOError):
            print("Error reading existing JSON file; creating a new one.")
            existing_data = new_timeseries
    else:
        # If the file doesn't exist, start with the new timeseries data
        existing_data = new_timeseries

    # Write back to the file
    with open(filename, "w") as file:
        json.dump(existing_data, file, indent=4)
    print(f"Data successfully saved to {filename}")

# Function to save the full YR JSON to a timestamped backup file
def save_full_json(yr_data):
    # Create the backup directory if it doesn't exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # Generate the backup filename
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_filename = os.path.join(backup_dir, f"yr_{timestamp}.json")

    # Save the full JSON to the backup file
    with open(backup_filename, "w") as backup_file:
        json.dump(yr_data, backup_file, indent=4)
    print(f"Full YR data saved to {backup_filename}")

# Main script
if __name__ == "__main__":
    print(f"Fetching weather data for {location} using YR.no API...")

    # Try fetching data from version 2.0
    yr_data, headers = fetch_weather(latitude, longitude, altitude, use_v2=True)
    if not yr_data:
        print("Version 2.0 failed, falling back to version 1.0")
        yr_data, headers = fetch_weather(latitude, longitude, altitude, use_v2=False)

    if yr_data:
        # Save the full JSON to a timestamped backup file
        save_full_json(yr_data)

        # Calculate solar angles
        solar_data = calculate_solar_angles(latitude, longitude)

        # Combine data
        combined_data = {
            "location": location,
            "yr_data": yr_data,
            "solar_data": solar_data,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "expires": headers.get("Expires") if headers else None,
        }

        # Save or append timeseries data to JSON
        save_or_append_json(combined_data, output_file)

        # Respect the 'Expires' header to avoid frequent requests
        if headers and headers.get("Expires"):
            expires_time = datetime.strptime(headers["Expires"], "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
            sleep_duration = (expires_time - datetime.now(timezone.utc)).total_seconds()
            print(f"Sleeping for {max(0, sleep_duration)} seconds to respect API rate limits.")
            sleep(max(0, sleep_duration))
    else:
        print("Failed to fetch weather data.")

