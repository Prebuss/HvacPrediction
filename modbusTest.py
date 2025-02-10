#!/usr/bin/env python3

import minimalmodbus
import serial

def main():
    # Configure the Modbus instrument
    # Replace '/dev/ttyUSB0' with the correct serial port for your RS485 dongle
    instrument = minimalmodbus.Instrument('/dev/ttyUSB0', 1)  # port name, slave address (1 here is an example)
    instrument.serial.baudrate = 9600
    instrument.serial.bytesize = 8
    instrument.serial.parity   = serial.PARITY_NONE
    instrument.serial.stopbits = 1
    instrument.serial.timeout  = 1  # seconds
    instrument.mode = minimalmodbus.MODE_RTU

    try:
        # 3xXXXX => input registers => function code = 4 in minimalmodbus
        co2_sensor_data              = instrument.read_register(267, 0, 4)   # 3x0267
        outdoor_temperature          = instrument.read_register(73,  0, 4)   # 3x0073
        supply_air_flow_pressure     = instrument.read_register(30,  0, 4)   # 3x0030
        supply_air_flow              = instrument.read_register(6,   0, 4)   # 3x0006
        ahu_filter_pressure_level    = instrument.read_register(27,  0, 4)   # 3x0027
        fan_levels                   = instrument.read_register(38,  0, 4)   # 3x0038
        reheat_regulation_level      = instrument.read_register(94,  0, 4)   # 3x0094
        cooling_regulation_level     = instrument.read_register(101, 0, 4)   # 3x0101
        duct_pressure                = instrument.read_register(8,   0, 4)   # 3x0008
        voc_level                    = instrument.read_register(266, 0, 4)   # 3x0266
        rhx_operation_level          = instrument.read_register(105, 0, 4)   # 3x0105
        rhx_efficiency               = instrument.read_register(106, 0, 4)   # 3x0106
        rhx_defrost_pressure_level   = instrument.read_register(107, 0, 4)   # 3x0107
        heat_exchanger_regulator_lvl = instrument.read_register(91,  0, 4)   # 3x0091

        # 4xXXXX => holding registers => function code = 3 in minimalmodbus
        room_temp_set_point          = instrument.read_register(135, 0, 3)   # 4x0135
        min_temperature_set_point    = instrument.read_register(136, 0, 3)   # 4x0136
        max_temperature_set_point    = instrument.read_register(137, 0, 3)   # 4x0137

        # Print out results
        print("CO? sensor data:                 ", co2_sensor_data)
        print("Outdoor temperature:             ", outdoor_temperature)
        print("Supply air flow pressure:        ", supply_air_flow_pressure)
        print("Supply air flow:                 ", supply_air_flow)
        print("AHU filter pressure level:       ", ahu_filter_pressure_level)
        print("Fan levels:                      ", fan_levels)
        print("Reheat regulation level:         ", reheat_regulation_level)
        print("Cooling regulation level:        ", cooling_regulation_level)
        print("Duct pressure:                   ", duct_pressure)
        print("VOC level:                       ", voc_level)
        print("RHX operation level:             ", rhx_operation_level)
        print("RHX efficiency:                  ", rhx_efficiency)
        print("RHX defrost pressure level:      ", rhx_defrost_pressure_level)
        print("Heat exchanger regulator level:  ", heat_exchanger_regulator_lvl)
        print("Room temperature set point:      ", room_temp_set_point)
        print("Minimum temperature set point:   ", min_temperature_set_point)
        print("Maximum temperature set point:   ", max_temperature_set_point)

    except Exception as e:
        print(f"An error occurred while reading Modbus registers: {e}")


if __name__ == "__main__":
    main()

