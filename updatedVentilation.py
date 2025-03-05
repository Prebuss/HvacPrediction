#!/usr/bin/env python3
"""
This script is intended to be run from cron.
It:
  - Reads CO2 values from an Arduino via HTTP (JSON format)
  - Reads ventilation parameters via Modbus registers using minimalmodbus
  - Appends the data (with timestamp) to a JSON file stored in the 'ventilation/' folder.
  - The output filename is ventilationYYYY-MM-DD.json (one file per day).
"""

import os
import json
import datetime
import requests
import minimalmodbus

def fetch_arduino_data(url="http://10.205.104.247"):
    """
    Fetch JSON data from the Arduino.
    The Arduino is expected to return a JSON object like: {"co2":446}
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # will raise an exception for HTTP errors
        return response.json()
    except Exception as e:
        # In case of error, return a dict with error info
        return {"arduino_error": f"Error fetching Arduino data: {e}"}

def fetch_modbus_data():
    """
    Connect to the ventilation system via Modbus and read various registers.
    Adjust the port, slave address, and serial settings as needed.
    """
    data = {}
    try:
        # Initialize the instrument.
        # Adjust '/dev/ttyUSB0' and slave address (1) as needed.
        instrument = minimalmodbus.Instrument('/dev/ttyUSB0', 1)
        instrument.serial.baudrate = 9600       # Baud rate
        instrument.serial.bytesize = 8
        instrument.serial.parity   = minimalmodbus.serial.PARITY_NONE
        instrument.serial.stopbits = 1
        instrument.serial.timeout  = 1            # seconds

        # Read input registers using function code 4
        data["supplyAirflow"]                = instrument.read_register(0,   0, 4) # 3x0001
        data["extractAirflow"]               = instrument.read_register(2,   0, 4) # 3x0003
        data["supplyAirDuctPressure"]        = instrument.read_register(4,   0, 4) # 3x0005
        data["extractAirDuctPressure"]       = instrument.read_register(6,   0, 4) # 3x0007
        data["supplyAirFanSpeedLevel"]       = instrument.read_register(12,  0, 4) # 3x0013
        data["extractAirFanSpeedLevel"]      = instrument.read_register(13,  0, 4) # 3x0014
        data["SFP"]                          = instrument.read_register(16,  0, 4) # 3x0017
        data["supplyAirTemperature"]         = instrument.read_register(27,  0, 4) # 3x0028
        data["extractAirTemperature"]        = instrument.read_register(28,  0, 4) # 3x0029
        data["outdoorTemperature"]           = instrument.read_register(29,  0, 4, signed = True) # 3x0030
        data["reheatLevel"]                  = instrument.read_register(36,  0, 4) # 3x0037
        data["coolingLevel"]                 = instrument.read_register(39,  0, 4) # 3x0040
        data["supplyAirFilterPressure"]      = instrument.read_register(49,  0, 4) # 3x0050
        data["extractAirFilterPressure"]     = instrument.read_register(52,  0, 4) # 3x0053
        data["operationMode"]                = instrument.read_register(84,  0, 4) # 3x0085
        data["heatExchangerRegulator"]       = instrument.read_register(90,  0, 4) # 3x0091
        data["RhxEfficiency"]                = instrument.read_register(105, 0, 4) # 3x0106

        # Read holding registers using function code 3
        data["setpointTemperatureSA"]        = instrument.read_register(40,  2, 3) # 4x0041
        data["setpointTemperatureEA"]        = instrument.read_register(41,  2, 3) # 4x0042
    except Exception as e:
        data["modbus_error"] = f"Error reading Modbus registers: {e}"
    return data

def append_data_to_file(record, folder="ventilation"):
    """
    Save or append the record (a dict) to a JSON file in the given folder.
    The filename is based on today's date.
    """
    # Ensure the folder exists.
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create filename using today's date.
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(folder, f"newVentilation{today_str}.json")

    # If file exists, load the existing data (assumed to be a list), then append the new record.
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = []
        except Exception:
            existing_data = []
    else:
        existing_data = []

    existing_data.append(record)

    # Write the updated list back to the file.
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)

def main():
    # Get the current timestamp in ISO format.
    timestamp = datetime.datetime.now().isoformat()

    # Create a record dictionary with the timestamp.
    record = {"timestamp": timestamp}

    # Fetch the Arduino data.
    arduino_data = fetch_arduino_data()
    record.update(arduino_data)

    # Fetch the Modbus data.
    modbus_data = fetch_modbus_data()
    record.update(modbus_data)

    # Append the record to the daily JSON file.
    append_data_to_file(record)

if __name__ == "__main__":
    main()
