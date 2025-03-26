#!/usr/bin/env python3
import configparser
import ssl
import paho.mqtt.client as mqtt

def main():
    # Load configuration from config.ini
    config = configparser.ConfigParser(interpolation=None)
    config.read('config.ini')

    # Get MQTT configuration values
    client_id = config.get('MQTT', 'client_id')
    host = config.get('MQTT', 'host')
    username = config.get('MQTT', 'username')
    password = config.get('MQTT', 'password')
    port = config.getint('MQTT', 'port')
    topic = config.get('MQTT', 'topic').strip()

    # Use a default topic if none is provided
    if not topic:
        topic = "default/topic"

    # Create a new MQTT client instance with the Node ID as client_id
    client = mqtt.Client(client_id)
    client.username_pw_set(username, password)

    # Configure TLS if using port 8883 (secure connection)
    if port == 8883:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)
        client.tls_insecure_set(False)

    # Define the on_connect callback function
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected successfully to the MQTT broker.")
            # Subscribe to the topic once connected
            client.subscribe(topic)
            print("Subscribed to topic:", topic)
        else:
            print("Failed to connect, return code:", rc)

    # Define the on_message callback function to process incoming messages
    def on_message(client, userdata, message):
        # Decode the payload from bytes to string (assuming UTF-8 encoding)
        payload = message.payload.decode("utf-8")
        print(f"Received message on topic '{message.topic}': {payload}")

    # Assign the callback functions to the client
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to the MQTT broker
    print(f"Connecting to MQTT broker at {host}:{port} ...")
    client.connect(host, port)

    # Start the network loop to listen for messages
    client.loop_forever()

if __name__ == "__main__":
    main()
