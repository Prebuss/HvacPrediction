from pysolar.solar import get_altitude, get_azimuth
from datetime import datetime, timedelta, timezone
import json

# Location: Solbråveien 43, Asker, Norway
latitude = 59.8338
longitude = 10.4376

# Define the start and end dates (half a year)
start_date = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
end_date = datetime(2025, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
time_interval = timedelta(hours=1)

# Output JSON file
output_file = "solar_angles_half_year.json"

# Calculate and save solar angles
def calculate_and_save_solar_angles():
    print("Calculating solar angles for half a year...")
    solar_data = []

    current_time = start_date
    while current_time < end_date:
        altitude = get_altitude(latitude, longitude, current_time)
        azimuth = get_azimuth(latitude, longitude, current_time)
        
        # Add the data to the list
        solar_data.append({
            "timestamp": current_time.isoformat(),
            "date": current_time.strftime("%Y-%m-%d"),
            "time": current_time.strftime("%H:%M:%S"),
            "solar_altitude": altitude,
            "solar_azimuth": azimuth
        })
        
        current_time += time_interval

    # Save to JSON file
    with open(output_file, "w") as json_file:
        json.dump(solar_data, json_file, indent=4)
    print(f"Solar angle data saved to {output_file}")

if __name__ == "__main__":
    calculate_and_save_solar_angles()
