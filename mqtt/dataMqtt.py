import configparser
import requests
import json

import paho.mqtt.client as mqtt

def read_co2_from_arduino(url):
    """
    Performs an HTTP GET request to the Arduino's WiFi endpoint,
    which should return a JSON object like {"co2": 642}.
    """
    try:
        response = requests.get(url, timeout=5)  # 5-second timeout
        response.raise_for_status()             # Raise an exception on HTTP errors
        data = response.json()
        return data  # e.g. {"co2": 642}
    except (requests.RequestException, ValueError) as e:
        print(f"Error reading CO2 data from Arduino: {e}")
        return {}

def publish_to_mqtt(data, mqtt_config):
    """
    Publish the collected data as JSON to the MQTT broker.
    """
    client = mqtt.Client(client_id=mqtt_config["client_id"])

    # If username/password are set, configure them
    if mqtt_config["username"] and mqtt_config["password"]:
        client.username_pw_set(mqtt_config["username"], mqtt_config["password"])

    # Connect to the MQTT broker
    client.connect(mqtt_config["host"], int(mqtt_config["port"]), 60)

    # Convert the Python dictionary to JSON
    payload = json.dumps(data)

    # Publish to the specified topic
    client.publish(mqtt_config["topic"], payload)

    # Disconnect from the broker
    client.disconnect()

def main():
    # Read configuration
    config = configparser.ConfigParser(interpolation = None)
    config.read("config.ini")

    # Arduino config
    arduino_url = config.get("ARDUINO", "url", fallback="http://10.205.104.247/")

    # MQTT config
    mqtt_config = {
        "host": config.get("MQTT", "host", fallback="localhost"),
        "port": config.get("MQTT", "port", fallback="1883"),
        "username": config.get("MQTT", "username", fallback=""),
        "password": config.get("MQTT", "password", fallback=""),
        "client_id": config.get("MQTT", "client_id", fallback="arduino_co2_client"),
        "topic": config.get("MQTT", "topic", fallback="arduino/co2")
    }

    # 1) Read CO? data from Arduino's WiFi endpoint
    co2_data = read_co2_from_arduino(arduino_url)

    # Optional: Print data locally for debugging
    print("Data read from Arduino:", co2_data)

    # 2) Publish that data (JSON) to MQTT
    if co2_data:
        publish_to_mqtt(co2_data, mqtt_config)
    else:
        print("No data to publish to MQTT (CO2 data was empty or error).")

if __name__ == "__main__":
    main()
