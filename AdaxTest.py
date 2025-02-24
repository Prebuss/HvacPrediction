#!/usr/bin/env python3

# install dependencies with: pip install requests sanction
import requests
import sanction
import datetime
import json
import os
import re

CLIENT_ID = "525951"   # replace with your client ID
CLIENT_SECRET = "vGN9LG0XhV861qt7"  # replace with your client secret
API_URL = "https://api-1.adax.no/client-api"

oauthClinet = sanction.Client(token_endpoint=API_URL + '/auth/token')


def get_token():
    """Authenticate and obtain JWT token."""
    oauthClinet.request_token(
        grant_type='password',
        username=CLIENT_ID,
        password=CLIENT_SECRET
    )
    return oauthClinet.access_token


def refresh_token():
    """Refresh JWT token."""
    oauthClinet.request_token(
        grant_type='refresh_token',
        refresh_token=oauthClinet.refresh_token,
        username=CLIENT_ID,
        password=CLIENT_SECRET
    )
    return oauthClinet.access_token


def extract_digits(text: str) -> str:
    """
    Returns the first group of consecutive digits found in 'text', or '' if none.
    E.g. "Office 409" -> "409"
         "Heater 310 Left" -> "310"
         "Meeting room 4th floor" -> '' (no digits)
    """
    match = re.findall(r'\d+', text)
    return match[0] if match else ''


def simplify_text(text: str) -> str:
    """
    Lowercase, strip, remove ' floor' suffix, etc.
    Example: "Meeting room 4th floor" -> "meeting room 4th"
    Adapt as needed for your naming patterns.
    """
    txt = text.lower().strip()
    # If ending with " floor", remove it
    if txt.endswith(" floor"):
        txt = txt[:-6].strip()  # remove ' floor' (6 chars)
    return txt


def belongs_to_room(room_name: str, device_name: str) -> bool:
    """
    Determine if 'device_name' matches 'room_name'.
    1. If the room has digits (e.g., "409"), and the device has digits (e.g. "409"),
       compare them directly.
    2. Else, do a partial substring check on simplified text.
    """
    # Extract digits from both
    rn_digits = extract_digits(room_name)
    dn_digits = extract_digits(device_name)

    if rn_digits and dn_digits:
        # If both have digits, compare them
        return rn_digits == dn_digits
    else:
        # Otherwise, fallback to partial substring match
        simple_room = simplify_text(room_name)
        simple_device = simplify_text(device_name)
        # Return True if the simplified room name is in device name
        # or vice versa. Adjust to your preference.
        return (simple_room in simple_device) or (simple_device in simple_room)


def get_rooms_and_devices(token):
    """
    Fetches data from the Adax API and:
      - Excludes the room named 'Yttergang'
      - Collects each room's temperature
      - Finds matching device(s) for each room using belongs_to_room()
    """
    headers = {"Authorization": "Bearer " + token}
    response = requests.get(API_URL + "/rest/v1/content/?withEnergy=1", headers=headers)
    response.raise_for_status()
    data = response.json()

    raw_rooms = data.get('rooms', [])
    raw_devices = data.get('devices', [])

    # 1) Create room list (skip Yttergang)
    rooms_data = []
    for room in raw_rooms:
        room_name = room.get('name', '')
        if room_name == "Yttergang":
            continue  # skip

        rooms_data.append({
            "id": room.get('id'),
            "name": room_name,
            "targetTemperature": room.get('targetTemperature', 0) / 100.0,
            "currentTemperature": room.get('temperature', 0) / 100.0,
            "devices": []
        })

    # 2) Match devices to rooms
    for device in raw_devices:
        device_name = device.get('name', '')
        # Also skip if device name is literally "Yttergang" (if you want to exclude that)
        if device_name.strip().lower() == "yttergang":
            continue

        device_info = {
            "id": device.get('id'),
            "name": device_name,
            "energy": device.get('energyWh', 0),
            "energyTime": datetime.datetime.utcfromtimestamp(
                int(device.get('energyTime', 0)) / 1000
            ).strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Attempt to find a matching room
        matched_any = False
        for room_entry in rooms_data:
            if belongs_to_room(room_entry["name"], device_name):
                room_entry["devices"].append(device_info)
                matched_any = True
                break

        # If a device didn't match any room, you could handle it here if you like:
        # if not matched_any:
        #     print(f"Unmatched device: {device_name}")

    return rooms_data


def save_data_to_json(rooms_data):
    """
    Appends the data to a daily JSON file in adaxTemp/ folder.
    The file is named YYYY-MM-DD.json.
    """
    folder_name = "adaxTemp"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(folder_name, f"{today_str}.json")

    # Prepare the new entry with a timestamp
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "timestamp": current_time_str,
        "rooms": rooms_data
    }

    # If the file exists, load existing data and append
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(new_entry)

    # Write updated data back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2)


def main():
    token = get_token()
    # Optional refresh if token is short-lived
    token = refresh_token()

    rooms_data = get_rooms_and_devices(token)
    save_data_to_json(rooms_data)


if __name__ == "__main__":
    main()
