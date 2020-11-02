import logging
import socket
import struct
import threading
import time


class WorkingThread(threading.Thread):
    def __init__(self, run_event, soc_bat, data, min_power, max_power, name):
        threading.Thread.__init__(self)
        self.run_event = run_event
        self.soc_bat = soc_bat
        self.data = data
        self.min_power = min_power
        self.max_power = max_power
        self.cur_power = 0.

        self.default_command = float(1)
        self.default_power = float(0)
        self.time_tic = 0.2
        self.name = name

    def update(self, power):
        self.cur_power = power

    def run(self):
        while self.run_event.is_set():
            packet = [self.default_command, self.cur_power]
            packet = bytearray(struct.pack('>{0}f'.format(len(packet)), *packet))
            time.sleep(self.time_tic)
            logging.info("Send {} command with power: {}".format(self.name, self.cur_power))
            self.soc_bat.send(packet)


def create_controller(host, port, log_file, name, min_power=-25., max_power=25.):
    host = host
    port = int(port)
    min_power = float(min_power)
    max_power = float(max_power)
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d  %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S')
    soc_bat = socket.socket(socket.AF_INET,  # Internet
                            socket.SOCK_DGRAM)  # UDP
    soc_bat.connect((host, port))

    run_event = threading.Event()

    run_event.set()

    return WorkingThread(run_event, soc_bat, [], min_power, max_power, name)
