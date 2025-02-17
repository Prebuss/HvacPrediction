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
    instrument.serial.timeout  = 10  # seconds
    instrument.mode = minimalmodbus.MODE_RTU

    try:
        # 3xXXXX => input registers => function code = 4 in minimalmodbus
        componentName                = instrument.read_register(0,   0, 4)   # 3x0001 
        co2_sensor_data              = instrument.read_register(267, 0, 4, signed=False)   # 3x0267
        co2_sensor_data2             = instrument.read_register(267, 0, 4, signed=True)   # 3x0267 Signed
        new_co2                      = instrument.read_register(266, 0, 4)   # 3x0267 But zero adjusted
        nothing                      = instrument.read_register(62,  0, 4)   # 3x0062 Should be zero
        SFP                          = instrument.read_register(63,  0, 4)   # 3x0063 Specific Fan Power
        SA_temp                      = instrument.read_register(64,  0, 4)   # 3x0064 Supply Air temperature
        outdoor_temperature          = instrument.read_register(73,  2, 4, signed=True)   # 3x0073
        supply_air_flow_pressure     = instrument.read_register(30,  0, 4)   # 3x0030
        supply_air_flow              = instrument.read_register(6,   0, 4)   # 3x0006
        ahu_filter_pressure_level    = instrument.read_register(27,  0, 4)   # 3x0027
        fan_levels                   = instrument.read_register(38,  0, 4)   # 3x0038
        reheat_regulation_level      = instrument.read_register(94,  0, 4)   # 3x0094
        cooling_regulation_level     = instrument.read_register(101, 0, 4)   # 3x0101
        duct_pressure                = instrument.read_register(8,   0, 4)   # 3x0008
        voc_level                    = instrument.read_register(266, 0, 4)   # 3x0266
        voc_level2                   = instrument.read_register(265, 0, 4)   # 3x0266 But zero adjusted
        rhx_operation_level          = instrument.read_register(105, 0, 4)   # 3x0105
        rhx_efficiency               = instrument.read_register(106, 0, 4)   # 3x0106
        rhx_defrost_pressure_level   = instrument.read_register(107, 0, 4)   # 3x0107
        heat_exchanger_regulator_lvl = instrument.read_register(91,  0, 4)   # 3x0091

        # 4xXXXX => holding registers => function code = 3 in minimalmodbus
        room_temp_set_point          = instrument.read_register(135, 0, 3)   # 4x0135
        min_temperature_set_point    = instrument.read_register(136, 0, 3)   # 4x0136
        max_temperature_set_point    = instrument.read_register(137, 0, 3)   # 4x0137

        # Test registers:
        AppID                        = instrument.read_register(1,   0, 3)   # 3x0002
        PHX23D                       = instrument.read_register(114, 0, 3)   # 3x0114
        XzoneCoil                    = instrument.read_register(197, 0, 3)   # 3x0198
        RhxOperationTime             = instrument.read_register(111, 0, 3)   # 3x0111 
        #OperationLevel               = instrument.read_register(592, 0, 4)   # 3x0592 - This was illegal
        #SoftwareVersion              = instrument.read_register(570, 0, 4)   # 3x0570

        # Print test:
        print("Application ID:                  ", AppID)
        print("PHX-2 3D Type:                   ", PHX23D)
        print("XZone Coil Type:                 ", XzoneCoil)
        print("RHX Operation Time:              ", RhxOperationTime)
        #print("Operation level:                 ", OperationLevel)
        #print("Software version:                ", SoftwareVersion)
        print("---------------------------------\n")

        # Print out results
        print("Component name:                  ", componentName)
        print("CO? sensor data:                 ", co2_sensor_data)
        print("Signed CO2 sensor data:          ", co2_sensor_data2)
        print("Co2 sensor data:                 ", new_co2)
        print("This should be zero:             ", nothing)
        print("SFP:                             ", SFP)
        print("Supply air temperature           ", SA_temp)
        print("Outdoor temperature:             ", outdoor_temperature)
        print("Supply air flow pressure:        ", supply_air_flow_pressure)
        print("Supply air flow:                 ", supply_air_flow)
        print("AHU filter pressure level:       ", ahu_filter_pressure_level)
        print("Fan levels:                      ", fan_levels)
        print("Reheat regulation level:         ", reheat_regulation_level)
        print("Cooling regulation level:        ", cooling_regulation_level)
        print("Duct pressure:                   ", duct_pressure)
        print("VOC level:                       ", voc_level)
        print("VOC level zero adjusted:         ", voc_level2)
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

