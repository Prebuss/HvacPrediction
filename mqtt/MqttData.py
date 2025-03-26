#!/usr/bin/env python3

import configparser
import minimalmodbus
import serial
import requests
import json
import time

import paho.mqtt.client as mqtt

###############################################################################
# 1) Read Arduino CO? data over HTTP (JSON)
###############################################################################
def read_co2_from_arduino(url):
    """
    Sends an HTTP GET request to the Arduino's endpoint (e.g. http://10.205.104.247/)
    which returns something like {"co2": 594}.
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()  # e.g. {"co2": 615}
    except (requests.RequestException, ValueError) as e:
        print(f"Error reading CO2 data from Arduino: {e}")
        return {}

###############################################################################
# 2) Read Modbus RTU data over RS485
###############################################################################
def read_modbus_data(port_name, slave_id=1, baudrate=9600):
    """
    Uses minimalmodbus to read the registers you specified:
      3x0267  -> co2_sensor_data
      4x0135  -> room_temp_set_point, etc.

    NOTE: This requires an RS485 interface at port_name (e.g. /dev/ttyUSB0).
    """
    data = {}
    instrument = minimalmodbus.Instrument(port_name, slave_id)  # port, slave address
    instrument.serial.baudrate = baudrate
    instrument.serial.bytesize = 8
    instrument.serial.parity   = serial.PARITY_NONE
    instrument.serial.stopbits = 1
    instrument.serial.timeout  = 1  # seconds
    instrument.mode = minimalmodbus.MODE_RTU

    try:
        # 3xXXXX => Function code 4 in minimalmodbus (read input registers)
        data["co2_sensor_data"]              = instrument.read_register(267, 0, 4)
        data["outdoor_temperature"]          = instrument.read_register(73,  0, 4)
        data["supply_air_flow_pressure"]     = instrument.read_register(30,  0, 4)
        data["supply_air_flow"]              = instrument.read_register(6,   0, 4)
        data["ahu_filter_pressure_level"]    = instrument.read_register(27,  0, 4)
        data["fan_levels"]                   = instrument.read_register(38,  0, 4)
        data["reheat_regulation_level"]      = instrument.read_register(94,  0, 4)
        data["cooling_regulation_level"]     = instrument.read_register(101, 0, 4)
        data["duct_pressure"]                = instrument.read_register(8,   0, 4)
        data["voc_level"]                    = instrument.read_register(266, 0, 4)
        data["rhx_operation_level"]          = instrument.read_register(105, 0, 4)
        data["rhx_efficiency"]               = instrument.read_register(106, 0, 4)
        data["rhx_defrost_pressure_level"]   = instrument.read_register(107, 0, 4)
        data["heat_exchanger_regulator_lvl"] = instrument.read_register(91,  0, 4)

        # 4xXXXX => Function code 3 in minimalmodbus (read holding registers)
        data["room_temp_set_point"]          = instrument.read_register(135, 0, 3)
        data["min_temperature_set_point"]    = instrument.read_register(136, 0, 3)
        data["max_temperature_set_point"]    = instrument.read_register(137, 0, 3)

    except Exception as e:
        print(f"An error occurred while reading Modbus registers: {e}")

    return data

###############################################################################
# 3) Main script: read from Arduino + Modbus, publish to MQTT
#    with paho-mqtt "new" callback API (MQTTv5 signatures).
###############################################################################
def main():
    # Load config with interpolation disabled (so %2F is not misread).
    config = configparser.ConfigParser(interpolation=None)
    config.read("config.ini")

    # ARDUINO config
    arduino_url = config.get("ARDUINO", "url", fallback="http://10.205.104.247/")

    # MODBUS config
    port_name = config.get("MODBUS", "port_name", fallback="/dev/ttyUSB0")
    slave_id  = config.getint("MODBUS", "slave_id", fallback=1)
    baudrate  = config.getint("MODBUS", "baudrate", fallback=9600)

    # MQTT config
    mqtt_client_id = config.get("MQTT", "client_id", fallback="mqtt_client")
    mqtt_host      = config.get("MQTT", "host", fallback="localhost")
    mqtt_username  = config.get("MQTT", "username", fallback="")
    mqtt_password  = config.get("MQTT", "password", fallback="")
    mqtt_port      = config.getint("MQTT", "port", fallback=1883)
    mqtt_topic     = config.get("MQTT", "topic", fallback="test/topic")

    # 1) Read Arduino data
    arduino_data = read_co2_from_arduino(arduino_url)
    # 2) Read Modbus data
    modbus_data = read_modbus_data(port_name, slave_id, baudrate)

    # Combine them into a single dict
    combined_data = {
        "arduino": arduino_data,   # e.g. {"co2": 594}
        "modbus": modbus_data      # e.g. {...all modbus regs...}
    }

    print("Data to send:", combined_data)

    # 3) Publish to MQTT
    #    Using the v5 callback signatures to avoid the "Callback API version 1 is deprecated" warning
    client = mqtt.Client(
        client_id=mqtt_client_id,
        protocol=mqtt.MQTTv5
    )

    # If you need username/password (e.g., Azure IoT Hub)
    if mqtt_username and mqtt_password:
        client.username_pw_set(mqtt_username, mqtt_password)

    # MQTT v5 callbacks
    def on_connect(client, userdata, flags, reasonCode, properties=None):
        if reasonCode == 0:
            print("Connected to MQTT broker!")
        else:
            print(f"Failed to connect. reasonCode={reasonCode}")

    def on_disconnect(client, userdata, reasonCode, properties=None):
        print(f"Disconnected. Reason code: {reasonCode}")

    def on_message(client, userdata, message):
        print(f"Received MQTT message on {message.topic}: {message.payload.decode()}")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # If connecting to Azure IoT Hub with TLS on port 8883, you typically do:
    # import ssl
    # client.tls_set(
    #     ca_certs="path_to_azure_ca.crt",
    #     certfile=None,
    #     keyfile=None,
    #     cert_reqs=ssl.CERT_REQUIRED,
    #     tls_version=ssl.PROTOCOL_TLS,
    #     ciphers=None
    # )

    # Connect
    client.connect(mqtt_host, mqtt_port, keepalive=60)
    client.loop_start()  # Start background thread for network handling

    # Publish the JSON
    payload = json.dumps(combined_data)
    result = client.publish(mqtt_topic, payload)
    # Optionally check result.rc or result.wait_for_publish()

    time.sleep(2)  # Wait a moment to ensure publish is sent
    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    main()
