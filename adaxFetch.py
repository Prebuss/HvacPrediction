# install dependencies with: pip install requests sanction
import requests
import sanction
import time
import datetime

CLIENT_ID = "525951"  # replace with your client ID (see Adax WiFi app, Account Section)
CLIENT_SECRET = "vGN9LG0XhV861qt7"  # replace with your client secret (see Adax WiFi app, Account Section)
API_URL = "https://api-1.adax.no/client-api"

oauthClinet = sanction.Client(token_endpoint=API_URL + '/auth/token')


def get_token():
    # Authenticate and obtain JWT token
    oauthClinet.request_token(grant_type='password', username=CLIENT_ID, password=CLIENT_SECRET)
    return oauthClinet.access_token


def refresh_token():
    oauthClinet.request_token(grant_type='refresh_token', refresh_token=oauthClinet.refresh_token, username=CLIENT_ID, password=CLIENT_SECRET)
    return oauthClinet.access_token


def set_room_target_temperature(roomId, temperature, token):
    # Sets target temperature of the room
    headers = {"Authorization": "Bearer " + token}
    json = {'rooms': [{'id': roomId, 'targetTemperature': str(temperature)}]}
    requests.post(API_URL + '/rest/v1/control/', json=json, headers=headers)


def get_energy_info(token, roomId):
    headers = {"Authorization": "Bearer " + token}
    response = requests.get(API_URL + "/rest/v1/energy_log/" + str(roomId), headers=headers)
    
    # Check the response status
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response text: {response.text}")
        token = get_token()
        print(f"Generated Token: {token}")
        return

    try:
        json = response.json()
    except requests.exceptions.JSONDecodeError:
        print("Error: Unable to parse JSON response.")
        print(f"Response text: {response.text}")
        return

    for log in json['points']:
        fromTime = datetime.datetime.utcfromtimestamp(int(log['fromTime']) / 1000).strftime('%Y-%m-%d %H:%M:%S')
        toTime = datetime.datetime.utcfromtimestamp(int(log['toTime']) / 1000).strftime('%Y-%m-%d %H:%M:%S')
        energy = log['energyWh']
        print("From: %15s, To: %15s, %5dwh" % (fromTime, toTime, energy))


def get_homes_info(token):
    headers = {"Authorization": "Bearer " + token}
    response = requests.get(API_URL + "/rest/v1/content/?withEnergy=1", headers=headers)
    json = response.json()

    for room in json['rooms']:
        roomName = room['name']
        targetTemperature = room.get('targetTemperature', 0) / 100.0
        currentTemperature = room.get('temperature', 0) / 100.0
        print("Room: %15s, Target: %5.2fC, Temperature: %5.2fC, id: %5d" % (roomName, targetTemperature, currentTemperature, room['id']))

    if 'devices' in json:
        for device in json['devices']:
            deviceName = device['name']
            energy = device['energyWh']
            energyTime = datetime.datetime.utcfromtimestamp(int(device['energyTime']) / 1000).strftime('%Y-%m-%d %H:%M:%S')
            print("Device: %15s, Time: %15s, Energy: %5dwh, id: %5d" % (deviceName, energyTime, energy, device['id']))


token = get_token()

while True:
    #time.sleep(10)
    # Change the temperature to 24 C in the room with an Id of 735362
    #set_room_target_temperature(735362, 2400, token)

    # Replace the 735362 with the room id from the get_homes_info output
    #time.sleep(10)
    #set_room_target_temperature(750269, 2400, token)

    # Replace the 750269 with the room id from the get_homes_info output
    #time.sleep(10)
    get_homes_info(token)
    time.sleep(10)
    #get_energy_info(token,2429797)
    token = refresh_token()
