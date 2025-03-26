#!/usr/bin/env python3

import configparser
import minimalmodbus
import serial
import requests
import json
import time
import ssl

import paho.mqtt.client as mqtt

###############################################################################
# 1) Read Arduino CO? data via HTTP (JSON)
###############################################################################
def read_co2_from_arduino(url):
    """
    Sends an HTTP GET request to the Arduino's endpoint, e.g. http://10.205.104.247/
    which returns something like {"co2": 664}.
    """
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()  # e.g. {"co2": 664}
    except (requests.RequestException, ValueError) as e:
        print(f"Error reading CO? data from Arduino: {e}")
        return {}

###############################################################################
# 2) Read Modbus RTU data using minimalmodbus (serial)
###############################################################################
def read_modbus_data(port_name, slave_id=1, baudrate=9600):
    """
    Uses minimalmodbus to read your list of registers:
      - 3x0267 => function code 4
      - 4x0135 => function code 3
    Returns a dict of register values.
    """
    data = {}

    instrument = minimalmodbus.Instrument(port_name, slave_id)
    instrument.serial.baudrate = baudrate
    instrument.serial.bytesize = 8
    instrument.serial.parity   = serial.PARITY_NONE
    instrument.serial.stopbits = 1
    instrument.serial.timeout  = 1
    instrument.mode = minimalmodbus.MODE_RTU

    try:
        # 3x registers => function code 4
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

        # 4x registers => function code 3
        data["room_temp_set_point"]          = instrument.read_register(135, 0, 3)
        data["min_temperature_set_point"]    = instrument.read_register(136, 0, 3)
        data["max_temperature_set_point"]    = instrument.read_register(137, 0, 3)

    except Exception as e:
        print(f"Error reading Modbus registers: {e}")

    return data

###############################################################################
# 3) Main script: read from Arduino & Modbus, publish to Azure IoT (MQTT v5 + TLS)
###############################################################################
def main():
    # Use interpolation=None so '%' in SAS token won't break configparser
    config = configparser.ConfigParser(interpolation=None)
    config.read("config.ini")

    # ARDUINO
    arduino_url = config.get("ARDUINO", "url", fallback="http://10.205.104.247/")

    # MODBUS
    port_name = config.get("MODBUS", "port_name", fallback="/dev/ttyUSB0")
    slave_id  = config.getint("MODBUS", "slave_id", fallback=1)
    baudrate  = config.getint("MODBUS", "baudrate", fallback=9600)

    # MQTT (Azure IoT Hub)
    mqtt_client_id = config.get("MQTT", "client_id", fallback="deviceId")
    mqtt_host      = config.get("MQTT", "host", fallback="yourHub.azure-devices.net")
    mqtt_username  = config.get("MQTT", "username", fallback="")
    mqtt_password  = config.get("MQTT", "password", fallback="")
    mqtt_port      = config.getint("MQTT", "port", fallback=8883)
    mqtt_topic     = config.get("MQTT", "topic", fallback="devices/deviceId/messages/events/")

    # 1) Gather data from Arduino + Modbus
    arduino_data = read_co2_from_arduino(arduino_url)
    modbus_data  = read_modbus_data(port_name, slave_id, baudrate)

    combined_data = {
        "arduino": arduino_data,
        "modbus": modbus_data
    }
    print("Data to send:", combined_data)

    # 2) Create an MQTT client using v2 callback API & MQTTv5 protocol
    client = mqtt.Client(
        #callback_api = 2,   # <--- Force v2 callbacks to remove "v1 is deprecated" warning
        client_id=mqtt_client_id,
        protocol=mqtt.MQTTv311
    )

    # Azure IoT Hub credentials
    if mqtt_username and mqtt_password:
        client.username_pw_set(mqtt_username, mqtt_password)

    # Define v5 callbacks using new signatures
    def on_connect(client, userdata, flags, reasonCode, properties=None):
        if reasonCode == 0:
            print("Connected to MQTT broker!")
        else:
            print(f"Failed to connect. reasonCode={reasonCode}")

    def on_disconnect(client, userdata, reasonCode, properties=None):
        print(f"Disconnected. Reason code: {reasonCode}")

    def on_message(client, userdata, message):
        print(f"Received on topic {message.topic}: {message.payload.decode()}")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # 3) Enable TLS (Azure IoT requires it on port 8883).
    #    Use an SSLContext with PROTOCOL_TLS_CLIENT (not PROTOCOL_TLSv1_2).
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    # For a production scenario, load Azure's official root cert:
    # context.load_verify_locations("path/to/BaltimoreCyberTrustRoot.crt.pem")
    # context.check_hostname = True
    # context.verify_mode = ssl.CERT_REQUIRED

    # If you do not have the CA file, you can do an insecure test:
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    client.tls_set_context(context)

    # 4) Connect & publish
    client.connect(mqtt_host, mqtt_port, keepalive=60)
    client.loop_start()

    payload = json.dumps(combined_data)
    result = client.publish(mqtt_topic, payload)

    # Wait briefly so we can observe the connection & publish
    time.sleep(2)
    client.loop_stop()
    client.disconnect()
    client.enable_logger()

if __name__ == "__main__":
    main()
