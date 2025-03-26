#!/usr/bin/env python3
import configparser
import ssl
import json
import paho.mqtt.client as mqtt
import requests
import time

def main():
    # Load configuration from config.ini
    config = configparser.ConfigParser(interpolation=None)
    config.read('config.ini')

    # Get MQTT configuration
    client_id = config.get('MQTT', 'client_id')
    host = config.get('MQTT', 'host')
    username = config.get('MQTT', 'username')
    password = config.get('MQTT', 'password')
    port = config.getint('MQTT', 'port')
    topic = config.get('MQTT', 'topic').strip()
    if not topic:
        topic = "default/topic"  # Use a default topic if none is set

    # Get Arduino configuration
    arduino_url = config.get('ARDUINO', 'url').strip()

    # Create an MQTT client instance
    client = mqtt.Client(client_id)
    client.username_pw_set(username, password)

    # Set up TLS for secure communication (required for port 8883)
    if port == 8883:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)
        client.tls_insecure_set(False)

    # Define what to do once connected to the MQTT broker
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected successfully to MQTT broker.")
            # Attempt to fetch CO2 data from the Arduino
            try:
                response = requests.get(arduino_url, timeout=5)
                response.raise_for_status()
                # Assume the Arduino returns JSON data like: {"co2": 568}
                data = response.json()
                co2_value = data.get("co2")
                if co2_value is None:
                    print("Error: 'co2' key not found in Arduino response.")
                    return
            except Exception as e:
                print(f"Failed to fetch data from Arduino: {e}")
                return

            # Prepare the payload as JSON
            payload = json.dumps({"CO2": co2_value})
            result = client.publish(topic, payload)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"Successfully published CO2 value: {co2_value} to topic: {topic}")
            else:
                print(f"Failed to publish message to topic {topic}, error code: {result.rc}")
        else:
            print(f"Failed to connect, return code {rc}")

    # Assign the callback function
    client.on_connect = on_connect

    # Connect to the MQTT broker and start the network loop
    print(f"Connecting to MQTT broker at {host}:{port} ...")
    client.connect(host, port)
    client.loop_forever()

if __name__ == "__main__":
    main()
    #sleep(10)
