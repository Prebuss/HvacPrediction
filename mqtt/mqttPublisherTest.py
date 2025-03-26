#!/usr/bin/env python3
import configparser
import ssl
import paho.mqtt.client as mqtt

def main():
    # Load the configuration file
    config = configparser.ConfigParser(interpolation=None)
    config.read('config.ini')

    # Read MQTT settings from the config file
    client_id = config.get('MQTT', 'client_id')
    host = config.get('MQTT', 'host')
    username = config.get('MQTT', 'username')
    password = config.get('MQTT', 'password')
    port = config.getint('MQTT', 'port')
    topic = config.get('MQTT', 'topic').strip()

    # Use a default topic if none is provided in the config
    if not topic:
        topic = "dummy/topic"

    # Create a new MQTT client instance
    client = mqtt.Client(client_id)

    # Set the username and password for the connection
    client.username_pw_set(username, password)

    # Since we're using port 8883 (MQTT over TLS), configure TLS settings.
    if port == 8883:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)
        client.tls_insecure_set(False)

    # Define what to do once connected
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected successfully to MQTT Broker.")
            # Publish a dummy payload once connected
            dummy_payload = "dummy value"
            result = client.publish(topic, dummy_payload)
            status = result[0]
            if status == 0:
                print(f"Sent `{dummy_payload}` to topic `{topic}`")
            else:
                print(f"Failed to send message to topic {topic}")
        else:
            print(f"Failed to connect, return code {rc}")

    # Bind the on_connect callback function to the client
    client.on_connect = on_connect

    # Connect to the MQTT broker
    print(f"Connecting to MQTT broker at {host}:{port} ...")
    client.connect(host, port)

    # Start the network loop and block forever
    client.loop_forever()

if __name__ == "__main__":
    main()
