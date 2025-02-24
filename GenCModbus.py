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
        #
        supplyAirflow                = instrument.read_register(0,   0, 4)   # 3x0001
        extractAirflow               = instrument.read_register(2,   0, 4)   # 3x0003
        supplyAirDuctPressure        = instrument.read_register(4,   0, 4)   # 3x0005
        extractAirDuctPressure       = instrument.read_register(6,   0, 4)   # 3x0007
        supplyAirFanSpeedLevel       = instrument.read_register(12,  0, 4)   # 3x0013
        extractAirFanSpeedLevel      = instrument.read_register(13,  0, 4)   # 3x0014
        SFP                          = instrument.read_register(16,  1, 4)   #3x0017
        supplyAirTemperature         = instrument.read_register(27,  0, 4)   # 3x0028
        extractAirTemperature        = instrument.read_register(28,  0, 4)   # 3x0029
        outdoorTemperature           = instrument.read_register(29,  2, 4, signed=True)   # 3x0030
        reheatLevel                  = instrument.read_register(36,  2, 4)   # 3x0037
        coolingLevel                 = instrument.read_register(39,  2, 4)   # 3x0040
        supplyAirFilterPressure      = instrument.read_register(49,  0, 4)   # 3x0050
        extractAirFilterPressure     = instrument.read_register(52,  0, 4)   # 3x0053
        operationMode                = instrument.read_register(84,  0, 4)   # 3x0085
        heatExchangerRegulatorLevel  = instrument.read_register(90,  0, 4)   # 3x0091
        rhxDefrostPressureLevel      = instrument.read_register(105, 0, 4)   # 3x0107 

        # Print out results
        print("Supply Airflow:                  ", supplyAirflow, "\tl/s")
        print("Extract Airflow:                 ", extractAirflow, "\tl/s")
        print("Supply Air Duct Pressure:        ", supplyAirDuctPressure, "\tPa")
        print("Extract Air Duct Pressure:       ", extractAirDuctPressure, "\tPa")
        print("Supply Air Fan Speed Level:      ", supplyAirFanSpeedLevel, "\t%")
        print("Extract Air Fan Speed Level:     ", extractAirFanSpeedLevel, "\t%")
        print("Supply Air Temperature:          ", supplyAirTemperature, "\tC")
        print("Extract Air Temperature:         ", extractAirTemperature, "\tC")
        print("Outdoor Temperature:             ", outdoorTemperature, "C")
        print("Reheat Level:                    ", reheatLevel, "\t%")
        print("Cooling Level:                   ", coolingLevel, "\t%")
        print("Supply Air Filter Pressure:      ", supplyAirFilterPressure, "\tPa")
        print("Operation Mode:                  ", operationMode)
        print("Extract Air Filter Pressure:     ", extractAirFilterPressure, "\tPa")
        print("Heat Exchanger Regulator Level:  ", heatExchangerRegulatorLevel)
        print("RHX Defrost Pressure Level:      ", rhxDefrostPressureLevel, "\tPa")

    except Exception as e:
        print(f"An error occurred while reading Modbus registers: {e}")


if __name__ == "__main__":
    main()

