import paho.mqtt.client as mqtt
import configparser

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read("config.ini")

# Get MQTT credentials from config file
CLIENT_ID = config["MQTT"].get("client_id", "default_client_id")
HOST = config["MQTT"].get("host", "mqtt.example.com")
USERNAME = config["MQTT"].get("username", "your_username")
PASSWORD = config["MQTT"].get("password", "your_password")
PORT = config["MQTT"].getint("port", 1883)  # Default to 1883 if not provided
TOPIC = config["MQTT"].get("topic", "test/topic")

# Callback when the client connects to the broker
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(TOPIC)  # Subscribe to a topic
    else:
        print(f"Failed to connect, reason code {reason_code}")

# Callback for when a message is received
def on_message(client, userdata, message):
    print(f"Received `{message.payload.decode()}` from `{message.topic}` topic")

# Create MQTT client instance with Callback API v2
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, CLIENT_ID)
client.username_pw_set(USERNAME, PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

# Connect to broker
client.connect(HOST, PORT, 60)

# Start loop to process messages
client.loop_forever()
