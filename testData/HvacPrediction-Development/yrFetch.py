import requests
import json
from datetime import datetime, timezone
from time import sleep

# Define the location and output file
location = "Solbråveien 43, Asker, Norway"
latitude = 59.8338
longitude = 10.4376
altitude = 150  # Approximate altitude in meters for Solbråveien 43
output_file = "yr_weather_data.json"

# Define the YR API endpoint and headers
yr_api_url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
headers = {
    "User-Agent": "WeatherScript/1.0 (https://github.com/yourusername/WeatherScript)"  # Replace with your project's URL or name
}

# Function to fetch weather data from YR API
def fetch_yr_weather(lat, lon, alt):
    params = {"lat": lat, "lon": lon, "altitude": alt}
    try:
        response = requests.get(yr_api_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json(), response.headers
    except requests.exceptions.RequestException as e:
        print(f"Error fetching YR.no weather data: {e}")
        return None, None

# Save the data to a JSON file
def save_to_json(data, filename):
    try:
        with open(filename, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Weather data saved to {filename}")
    except IOError as e:
        print(f"Error saving data to JSON file: {e}")

# Main script
if __name__ == "__main__":
    print(f"Fetching weather data for {location} using YR.no API...")

    # Check if cached data exists and is still valid
    try:
        with open(output_file, "r") as json_file:
            existing_data = json.load(json_file)
            expires = existing_data.get("expires")
            if expires:
                # Convert 'Expires' to a datetime object
                expires_time = datetime.strptime(expires, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) < expires_time:
                    print("Using cached data; not fetching new data.")
                    exit()
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        # If no valid cached data is found, proceed to fetch new data
        pass

    # Fetch new data from YR.no
    yr_data, headers = fetch_yr_weather(latitude, longitude, altitude)
    if yr_data and headers:
        # Combine data
        combined_data = {
            "location": location,
            "yr_data": yr_data,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "expires": headers.get("Expires")  # Store the 'Expires' header for future use
        }

        # Save the new data to JSON
        save_to_json(combined_data, output_file)

        # Handle the 'Expires' header to avoid frequent requests
        if headers.get("Expires"):
            try:
                expires_time = datetime.strptime(headers["Expires"], "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
                sleep_duration = (expires_time - datetime.now(timezone.utc)).total_seconds()
                print(f"Sleeping for {sleep_duration} seconds to respect API rate limits.")
                sleep(max(0, sleep_duration))  # Ensure no negative sleep duration
            except ValueError:
                print("Failed to parse 'Expires' header; skipping sleep.")
    else:
        print("Failed to fetch new weather data; using existing data if available.")
