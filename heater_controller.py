import logging

import minimalmodbus
import serial


class HeaterController:
    def __init__(self, port, address, baudrate=9600, bytesize=8, parity=serial.PARITY_EVEN, stopbits=1, timeout=0.5,
                 mode=minimalmodbus.MODE_RTU, log_folder="log", log_name="heater"):
        # logger
        if log_name == "heater":
            log_name = "heater" + str(address)
        handler = logging.FileHandler(log_folder + "/" + log_name + ".log")
        formatter = logging.Formatter('%(asctime)s  %(message)s')
        handler.setFormatter(formatter)
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

        self.instrument = minimalmodbus.Instrument(port, address)
        self.instrument.serial.baudrate = baudrate
        self.instrument.serial.bytesize = bytesize
        self.instrument.serial.parity = parity
        self.instrument.serial.stopbits = stopbits
        self.instrument.serial.timeout = timeout
        self.instrument.mode = mode

    def get_cur_power(self):
        cur_power = 0.
        try:
            cur_power = self.instrument.read_register(97)
            self.logger.info("current power: " + str(cur_power))
        except Exception as e:
            self.logger.error(e)

        return cur_power / 1000.

    def set_power(self, power):
        power = round(power * 100, 1)
        try:
            if not self.has_errors():
                self.instrument.write_register(96, power, 1, functioncode=6)
                self.logger.info("set power: " + str(power))
        except Exception as e:
            self.logger.error(e)

    def has_errors(self):
        try:
            ## Read errors count ##
            has_errors = bool(self.instrument.read_register(256, 1))  # Registernumber, number of decimals
            if has_errors:
                self.logger.error("Detect errors")
        except Exception as e:
            self.logger.error(e)


if __name__ == '__main__':
    conrtoller = HeaterController('/dev/ttyMI2', 3)
    print(conrtoller.has_errors())
