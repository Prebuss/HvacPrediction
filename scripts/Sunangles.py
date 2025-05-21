#!/usr/bin/env python3
import datetime
import os
import argparse
import pandas as pd
from pysolar.solar import get_altitude, get_azimuth

def calculate_sun_angles(latitude, longitude, dt):
    """
    Calculate the sun's altitude and azimuth angles for a given location and time.
    """
    altitude = get_altitude(latitude, longitude, dt)
    azimuth = get_azimuth(latitude, longitude, dt)
    return altitude, azimuth

def generate_day_sun_angles(latitude, longitude, date, interval_minutes=5):
    """
    Generate sun angles at given intervals for an entire day.
    """
    data = []
    dt = datetime.datetime(date.year, date.month, date.day, tzinfo=datetime.timezone.utc)
    end_dt = dt + datetime.timedelta(days=1)
    
    while dt < end_dt:
        altitude, azimuth = calculate_sun_angles(latitude, longitude, dt)
        data.append({
            "datetime": dt.isoformat(),
            "altitude": altitude,
            "azimuth": azimuth
        })
        dt += datetime.timedelta(minutes=interval_minutes)
    return data

def main():
    print("Script started...", flush=True)
    
    parser = argparse.ArgumentParser(
        description="Calculate sun angles for each day at 5-minute intervals."
    )
    parser.add_argument("--latitude", type=float, default=59.8309044,
                        help="Latitude in decimal degrees (default: 59.8309044)")
    parser.add_argument("--longitude", type=float, default=10.4113331,
                        help="Longitude in decimal degrees (default: 10.4113331)")
    parser.add_argument("--start-date", type=str, required=True,
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, required=True,
                        help="End date in YYYY-MM-DD format (inclusive)")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Folder to save CSV files")
    parser.add_argument("--interval", type=int, default=5,
                        help="Interval in minutes between calculations (default: 5)")
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Created output folder: {args.output_folder}", flush=True)
    else:
        print(f"Output folder exists: {args.output_folder}", flush=True)

    # Parse start and end dates
    try:
        start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    except ValueError as e:
        print("Error parsing dates. Please ensure the format is YYYY-MM-DD.", flush=True)
        raise e

    # Loop over each day in the date range
    current_date = start_date
    while current_date <= end_date:
        print(f"Calculating sun angles for {current_date}...", flush=True)
        data = generate_day_sun_angles(args.latitude, args.longitude, current_date, args.interval)
        df = pd.DataFrame(data)
        filename = os.path.join(args.output_folder, f"{current_date}.csv")
        df.to_csv(filename, index=False)
        print(f"Saved data for {current_date} to {filename}", flush=True)
        current_date += datetime.timedelta(days=1)

    print("All calculations complete.", flush=True)

if __name__ == "__main__":
    main()
