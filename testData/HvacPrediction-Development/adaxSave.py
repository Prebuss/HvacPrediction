import requests
import sanction
import datetime
import json
import os

CLIENT_ID = "525951"  # replace with your client ID
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


def simplify_room_name(room_name: str) -> str:
    """
    Example of simplifying a room name for matching:
    - Lowercase
    - Remove trailing ' floor' if present
    - Strip extra spaces
    """
    rn = room_name.lower().strip()
    # If you literally have '4th floor' or '3rd floor' at the end, remove ' floor'
    if rn.endswith(" floor"):
        rn = rn.replace(" floor", "")
    return rn.strip()


def belongs_to_room(room_name_simple: str, device_name: str) -> bool:
    """
    Returns True if the simplified room name appears in the device name.
    This is a naive partial match approach.
    Example:
      Room: 'Meeting room 4th floor' -> 'meeting room 4th'
      Device: 'Meeting room 4th West' -> 'meeting room 4th west'
      => partial match, so belongs to room.

    For offices:
      Room: 'Office 409' -> 'office 409'
      Device: 'Heater 409' -> 'heater 409'
      => '409' is in both => match.

    Adjust logic as needed for your naming conventions.
    """
    dn = device_name.lower().strip()
    # Return True if the simplified room string is contained in the device name
    return (room_name_simple in dn)


def get_rooms_and_devices(token):
    """
    Fetches data for rooms and devices from the Adax API.
    Returns a structure that:
      - Excludes the room named 'Yttergang'
      - Includes the room's temperature info
      - Maps any device whose name partially matches the simplified room name
        to that room's 'devices' list.
    """
    headers = {"Authorization": "Bearer " + token}
    response = requests.get(API_URL + "/rest/v1/content/?withEnergy=1", headers=headers)
    response.raise_for_status()
    data = response.json()

    raw_rooms = data.get('rooms', [])
    raw_devices = data.get('devices', [])

    # 1) Build a list of rooms (excluding Yttergang)
    rooms_data = []
    for room in raw_rooms:
        room_name = room.get('name')
        if room_name == "Yttergang":
            # skip Yttergang
            continue

        room_id = room.get('id')
        target_temp = room.get('targetTemperature', 0) / 100.0
        current_temp = room.get('temperature', 0) / 100.0

        rooms_data.append({
            "id": room_id,
            "name": room_name,
            "targetTemperature": target_temp,
            "currentTemperature": current_temp,
            "devices": []  # we will fill this later
        })

    # 2) For each device, find which room it belongs to (if any)
    for device in raw_devices:
        device_name = device.get('name')
        # We skip device "Yttergang" if you also want to exclude that
        if device_name.lower().strip() == "yttergang":
            continue

        device_energy = device.get('energyWh', 0)
        device_energy_time = datetime.datetime.utcfromtimestamp(
            int(device.get('energyTime', 0)) / 1000
        ).strftime('%Y-%m-%d %H:%M:%S')
        device_id = device.get('id')

        # Build the device dictionary
        device_dict = {
            "name": device_name,
            "energy": device_energy,
            "energyTime": device_energy_time,
            "id": device_id
        }

        # Try to find a matching room by partial string match
        matched = False
        for room_entry in rooms_data:
            room_name_simple = simplify_room_name(room_entry["name"])
            if belongs_to_room(room_name_simple, device_name):
                room_entry["devices"].append(device_dict)
                matched = True
                break

        # If you have a device that doesn't match any room name, you can decide
        # whether to put it in a "misc" list or ignore it entirely.
        # Example:
        # if not matched:
        #    print(f"WARNING: Device {device_name} not matched to any room.")

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

    # If the file exists, load existing data and append to it
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
    # 1. Get token and refresh (if needed).
    token = get_token()
    token = refresh_token()  # optional, if the token is short-lived

    # 2. Fetch data: rooms & devices, excluding 'Yttergang'.
    rooms_data = get_rooms_and_devices(token)

    # 3. Save (append) data to a daily JSON file.
    save_data_to_json(rooms_data)


if __name__ == "__main__":
    main()
